//===--- IncludeCleaner.cpp - Unused/Missing Headers Analysis ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncludeCleaner.h"
#include "Config.h"
#include "Diagnostics.h"
#include "Headers.h"
#include "ParsedAST.h"
#include "Preamble.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "URI.h"
#include "clang-include-cleaner/Analysis.h"
#include "clang-include-cleaner/Record.h"
#include "clang-include-cleaner/Types.h"
#include "support/Logger.h"
#include "support/Path.h"
#include "support/Trace.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/TemplateName.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Inclusions/HeaderIncludes.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include <cassert>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace clang {
namespace clangd {

static bool AnalyzeStdlib = false;
void setIncludeCleanerAnalyzesStdlib(bool B) { AnalyzeStdlib = B; }

namespace {

// Returns the range starting at '#' and ending at EOL. Escaped newlines are not
// handled.
clangd::Range getDiagnosticRange(llvm::StringRef Code, unsigned HashOffset) {
  clangd::Range Result;
  Result.end = Result.start = offsetToPosition(Code, HashOffset);

  // Span the warning until the EOL or EOF.
  Result.end.character +=
      lspLength(Code.drop_front(HashOffset).take_until([](char C) {
        return C == '\n' || C == '\r';
      }));
  return Result;
}

bool isFilteredByConfig(const Config &Cfg, llvm::StringRef HeaderPath) {
  // Convert the path to Unix slashes and try to match against the filter.
  llvm::SmallString<64> NormalizedPath(HeaderPath);
  llvm::sys::path::native(NormalizedPath, llvm::sys::path::Style::posix);
  for (auto &Filter : Cfg.Diagnostics.Includes.IgnoreHeader) {
    if (Filter(NormalizedPath))
      return true;
  }
  return false;
}

static bool mayConsiderUnused(const Inclusion &Inc, ParsedAST &AST,
                              const Config &Cfg,
                              const include_cleaner::PragmaIncludes *PI) {
  // FIXME(kirillbobyrev): We currently do not support the umbrella headers.
  // System headers are likely to be standard library headers.
  // Until we have good support for umbrella headers, don't warn about them.
  if (Inc.Written.front() == '<') {
    if (AnalyzeStdlib && tooling::stdlib::Header::named(Inc.Written))
      return true;
    return false;
  }
  assert(Inc.HeaderID);
  auto HID = static_cast<IncludeStructure::HeaderID>(*Inc.HeaderID);
  auto FE = AST.getSourceManager().getFileManager().getFileRef(
      AST.getIncludeStructure().getRealPath(HID));
  assert(FE);
  if (PI) {
    if (PI->shouldKeep(Inc.HashLine + 1))
      return false;
    // Check if main file is the public interface for a private header. If so we
    // shouldn't diagnose it as unused.
    if (auto PHeader = PI->getPublic(*FE); !PHeader.empty()) {
      PHeader = PHeader.trim("<>\"");
      // Since most private -> public mappings happen in a verbatim way, we
      // check textually here. This might go wrong in presence of symlinks or
      // header mappings. But that's not different than rest of the places.
      if (AST.tuPath().endswith(PHeader))
        return false;
    }
  }
  // Headers without include guards have side effects and are not
  // self-contained, skip them.
  if (!AST.getPreprocessor().getHeaderSearchInfo().isFileMultipleIncludeGuarded(
          &FE->getFileEntry())) {
    dlog("{0} doesn't have header guard and will not be considered unused",
         FE->getName());
    return false;
  }

  if (isFilteredByConfig(Cfg, Inc.Resolved)) {
    dlog("{0} header is filtered out by the configuration", FE->getName());
    return false;
  }
  return true;
}

llvm::StringRef getResolvedPath(const include_cleaner::Header &SymProvider) {
  switch (SymProvider.kind()) {
  case include_cleaner::Header::Physical:
    return SymProvider.physical()->tryGetRealPathName();
  case include_cleaner::Header::Standard:
    return SymProvider.standard().name().trim("<>\"");
  case include_cleaner::Header::Verbatim:
    return SymProvider.verbatim().trim("<>\"");
  }
  llvm_unreachable("Unknown header kind");
}

std::string getSymbolName(const include_cleaner::Symbol &Sym) {
  switch (Sym.kind()) {
  case include_cleaner::Symbol::Macro:
    return Sym.macro().Name->getName().str();
  case include_cleaner::Symbol::Declaration:
    return llvm::dyn_cast<NamedDecl>(&Sym.declaration())
        ->getQualifiedNameAsString();
  }
  llvm_unreachable("Unknown symbol kind");
}

std::vector<Diag> generateMissingIncludeDiagnostics(
    ParsedAST &AST, llvm::ArrayRef<MissingIncludeDiagInfo> MissingIncludes,
    llvm::StringRef Code) {
  std::vector<Diag> Result;
  const Config &Cfg = Config::current();
  if (Cfg.Diagnostics.MissingIncludes != Config::IncludesPolicy::Strict ||
      Cfg.Diagnostics.SuppressAll ||
      Cfg.Diagnostics.Suppress.contains("missing-includes")) {
    return Result;
  }

  const SourceManager &SM = AST.getSourceManager();
  const FileEntry *MainFile = SM.getFileEntryForID(SM.getMainFileID());

  auto FileStyle = format::getStyle(
      format::DefaultFormatStyle, AST.tuPath(), format::DefaultFallbackStyle,
      Code, &SM.getFileManager().getVirtualFileSystem());
  if (!FileStyle) {
    elog("Couldn't infer style", FileStyle.takeError());
    FileStyle = format::getLLVMStyle();
  }

  tooling::HeaderIncludes HeaderIncludes(AST.tuPath(), Code,
                                         FileStyle->IncludeStyle);
  for (const auto &SymbolWithMissingInclude : MissingIncludes) {
    llvm::StringRef ResolvedPath =
        getResolvedPath(SymbolWithMissingInclude.Providers.front());
    if (isFilteredByConfig(Cfg, ResolvedPath)) {
      dlog("IncludeCleaner: not diagnosing missing include {0}, filtered by "
           "config",
           ResolvedPath);
      continue;
    }

    std::string Spelling =
        spellHeader(AST, MainFile, SymbolWithMissingInclude.Providers.front());
    llvm::StringRef HeaderRef{Spelling};
    bool Angled = HeaderRef.starts_with("<");
    // We might suggest insertion of an existing include in edge cases, e.g.,
    // include is present in a PP-disabled region, or spelling of the header
    // turns out to be the same as one of the unresolved includes in the
    // main file.
    std::optional<tooling::Replacement> Replacement = HeaderIncludes.insert(
        HeaderRef.trim("\"<>"), Angled, tooling::IncludeDirective::Include);
    if (!Replacement.has_value())
      continue;

    Diag &D = Result.emplace_back();
    D.Message =
        llvm::formatv("No header providing \"{0}\" is directly included",
                      getSymbolName(SymbolWithMissingInclude.Symbol));
    D.Name = "missing-includes";
    D.Source = Diag::DiagSource::Clangd;
    D.File = AST.tuPath();
    D.InsideMainFile = true;
    D.Severity = DiagnosticsEngine::Warning;
    D.Range = clangd::Range{
        offsetToPosition(Code,
                         SymbolWithMissingInclude.SymRefRange.beginOffset()),
        offsetToPosition(Code,
                         SymbolWithMissingInclude.SymRefRange.endOffset())};
    auto &F = D.Fixes.emplace_back();
    F.Message = "#include " + Spelling;
    TextEdit Edit = replacementToEdit(Code, *Replacement);
    F.Edits.emplace_back(std::move(Edit));
  }
  return Result;
}

std::vector<Diag> generateUnusedIncludeDiagnostics(
    PathRef FileName, llvm::ArrayRef<const Inclusion *> UnusedIncludes,
    llvm::StringRef Code) {
  std::vector<Diag> Result;
  const Config &Cfg = Config::current();
  if (Cfg.Diagnostics.UnusedIncludes == Config::IncludesPolicy::None ||
      Cfg.Diagnostics.SuppressAll ||
      Cfg.Diagnostics.Suppress.contains("unused-includes")) {
    return Result;
  }
  for (const auto *Inc : UnusedIncludes) {
    Diag &D = Result.emplace_back();
    D.Message =
        llvm::formatv("included header {0} is not used directly",
                      llvm::sys::path::filename(
                          Inc->Written.substr(1, Inc->Written.size() - 2),
                          llvm::sys::path::Style::posix));
    D.Name = "unused-includes";
    D.Source = Diag::DiagSource::Clangd;
    D.File = FileName;
    D.InsideMainFile = true;
    D.Severity = DiagnosticsEngine::Warning;
    D.Tags.push_back(Unnecessary);
    D.Range = getDiagnosticRange(Code, Inc->HashOffset);
    // FIXME(kirillbobyrev): Removing inclusion might break the code if the
    // used headers are only reachable transitively through this one. Suggest
    // including them directly instead.
    // FIXME(kirillbobyrev): Add fix suggestion for adding IWYU pragmas
    // (keep/export) remove the warning once we support IWYU pragmas.
    auto &F = D.Fixes.emplace_back();
    F.Message = "remove #include directive";
    F.Edits.emplace_back();
    F.Edits.back().range.start.line = Inc->HashLine;
    F.Edits.back().range.end.line = Inc->HashLine + 1;
  }
  return Result;
}
} // namespace

std::vector<include_cleaner::SymbolReference>
collectMacroReferences(ParsedAST &AST) {
  const auto &SM = AST.getSourceManager();
  //  FIXME: !!this is a hacky way to collect macro references.
  std::vector<include_cleaner::SymbolReference> Macros;
  auto &PP = AST.getPreprocessor();
  for (const syntax::Token &Tok :
       AST.getTokens().spelledTokens(SM.getMainFileID())) {
    auto Macro = locateMacroAt(Tok, PP);
    if (!Macro)
      continue;
    if (auto DefLoc = Macro->Info->getDefinitionLoc(); DefLoc.isValid())
      Macros.push_back(
          {Tok.location(),
           include_cleaner::Macro{/*Name=*/PP.getIdentifierInfo(Tok.text(SM)),
                                  DefLoc},
           include_cleaner::RefType::Explicit});
  }
  return Macros;
}

include_cleaner::Includes
convertIncludes(const SourceManager &SM,
                const llvm::ArrayRef<Inclusion> Includes) {
  include_cleaner::Includes ConvertedIncludes;
  for (const Inclusion &Inc : Includes) {
    include_cleaner::Include TransformedInc;
    llvm::StringRef WrittenRef = llvm::StringRef(Inc.Written);
    TransformedInc.Spelled = WrittenRef.trim("\"<>");
    TransformedInc.HashLocation =
        SM.getComposedLoc(SM.getMainFileID(), Inc.HashOffset);
    TransformedInc.Line = Inc.HashLine + 1;
    TransformedInc.Angled = WrittenRef.starts_with("<");
    auto FE = SM.getFileManager().getFile(Inc.Resolved);
    if (!FE) {
      elog("IncludeCleaner: Failed to get an entry for resolved path {0}: {1}",
           Inc.Resolved, FE.getError().message());
      continue;
    }
    TransformedInc.Resolved = *FE;
    ConvertedIncludes.add(std::move(TransformedInc));
  }
  return ConvertedIncludes;
}

std::string spellHeader(ParsedAST &AST, const FileEntry *MainFile,
                        include_cleaner::Header Provider) {
  if (Provider.kind() == include_cleaner::Header::Physical) {
    if (auto CanonicalPath =
            getCanonicalPath(Provider.physical(), AST.getSourceManager())) {
      std::string SpelledHeader =
          llvm::cantFail(URI::includeSpelling(URI::create(*CanonicalPath)));
      if (!SpelledHeader.empty())
        return SpelledHeader;
    }
  }
  return include_cleaner::spellHeader(
      Provider, AST.getPreprocessor().getHeaderSearchInfo(), MainFile);
}

std::vector<const Inclusion *>
getUnused(ParsedAST &AST,
          const llvm::DenseSet<IncludeStructure::HeaderID> &ReferencedFiles,
          const llvm::StringSet<> &ReferencedPublicHeaders) {
  trace::Span Tracer("IncludeCleaner::getUnused");
  const Config &Cfg = Config::current();
  std::vector<const Inclusion *> Unused;
  for (const Inclusion &MFI : AST.getIncludeStructure().MainFileIncludes) {
    if (!MFI.HeaderID)
      continue;
    if (ReferencedPublicHeaders.contains(MFI.Written))
      continue;
    auto IncludeID = static_cast<IncludeStructure::HeaderID>(*MFI.HeaderID);
    bool Used = ReferencedFiles.contains(IncludeID);
    if (!Used && !mayConsiderUnused(MFI, AST, Cfg, AST.getPragmaIncludes())) {
      dlog("{0} was not used, but is not eligible to be diagnosed as unused",
           MFI.Written);
      continue;
    }
    if (!Used)
      Unused.push_back(&MFI);
    dlog("{0} is {1}", MFI.Written, Used ? "USED" : "UNUSED");
  }
  return Unused;
}

IncludeCleanerFindings computeIncludeCleanerFindings(ParsedAST &AST) {
  const auto &SM = AST.getSourceManager();
  const auto &Includes = AST.getIncludeStructure();
  include_cleaner::Includes ConvertedIncludes =
      convertIncludes(SM, Includes.MainFileIncludes);
  const FileEntry *MainFile = SM.getFileEntryForID(SM.getMainFileID());
  auto *PreamblePatch = PreamblePatch::getPatchEntry(AST.tuPath(), SM);

  std::vector<include_cleaner::SymbolReference> Macros =
      collectMacroReferences(AST);
  std::vector<MissingIncludeDiagInfo> MissingIncludes;
  llvm::DenseSet<IncludeStructure::HeaderID> Used;
  trace::Span Tracer("include_cleaner::walkUsed");
  include_cleaner::walkUsed(
      AST.getLocalTopLevelDecls(), /*MacroRefs=*/Macros,
      AST.getPragmaIncludes(), SM,
      [&](const include_cleaner::SymbolReference &Ref,
          llvm::ArrayRef<include_cleaner::Header> Providers) {
        bool Satisfied = false;
        for (const auto &H : Providers) {
          if (H.kind() == include_cleaner::Header::Physical &&
              (H.physical() == MainFile || H.physical() == PreamblePatch)) {
            Satisfied = true;
            continue;
          }
          for (auto *Inc : ConvertedIncludes.match(H)) {
            Satisfied = true;
            auto HeaderID = Includes.getID(Inc->Resolved);
            assert(HeaderID.has_value() &&
                   "ConvertedIncludes only contains resolved includes.");
            Used.insert(*HeaderID);
          }
        }

        if (Satisfied || Providers.empty() ||
            Ref.RT != include_cleaner::RefType::Explicit)
          return;

        // We actually always want to map usages to their spellings, but
        // spelling locations can point into preamble section. Using these
        // offsets could lead into crashes in presence of stale preambles. Hence
        // we use "getFileLoc" instead to make sure it always points into main
        // file.
        // FIXME: Use presumed locations to map such usages back to patched
        // locations safely.
        auto Loc = SM.getFileLoc(Ref.RefLocation);
        // File locations can be outside of the main file if macro is expanded
        // through an #include.
        while (SM.getFileID(Loc) != SM.getMainFileID())
          Loc = SM.getIncludeLoc(SM.getFileID(Loc));
        auto TouchingTokens =
            syntax::spelledTokensTouching(Loc, AST.getTokens());
        assert(!TouchingTokens.empty());
        // Loc points to the start offset of the ref token, here we use the last
        // element of the TouchingTokens, e.g. avoid getting the "::" for
        // "ns::^abc".
        MissingIncludeDiagInfo DiagInfo{
            Ref.Target, TouchingTokens.back().range(SM), Providers};
        MissingIncludes.push_back(std::move(DiagInfo));
      });
  std::vector<const Inclusion *> UnusedIncludes =
      getUnused(AST, Used, /*ReferencedPublicHeaders*/ {});
  return {std::move(UnusedIncludes), std::move(MissingIncludes)};
}

std::vector<Diag> issueIncludeCleanerDiagnostics(ParsedAST &AST,
                                                 llvm::StringRef Code) {
  // Interaction is only polished for C/CPP.
  if (AST.getLangOpts().ObjC)
    return {};

  trace::Span Tracer("IncludeCleaner::issueIncludeCleanerDiagnostics");

  const Config &Cfg = Config::current();
  IncludeCleanerFindings Findings;
  if (Cfg.Diagnostics.MissingIncludes == Config::IncludesPolicy::Strict ||
      Cfg.Diagnostics.UnusedIncludes == Config::IncludesPolicy::Strict) {
    // will need include-cleaner results, call it once
    Findings = computeIncludeCleanerFindings(AST);
  }

  std::vector<Diag> Result = generateUnusedIncludeDiagnostics(
      AST.tuPath(), Findings.UnusedIncludes, Code);
  llvm::move(
      generateMissingIncludeDiagnostics(AST, Findings.MissingIncludes, Code),
      std::back_inserter(Result));
  return Result;
}

} // namespace clangd
} // namespace clang
