//===-- ClangIncludeFixer.cpp - Standalone include fixer ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InMemorySymbolIndex.h"
#include "IncludeFixer.h"
#include "IncludeFixerContext.h"
#include "SymbolIndexManager.h"
#include "YamlSymbolIndex.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/YAMLTraits.h"

using namespace clang;
using namespace llvm;
using clang::include_fixer::IncludeFixerContext;

LLVM_YAML_IS_DOCUMENT_LIST_VECTOR(IncludeFixerContext)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(std::string)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(IncludeFixerContext::HeaderInfo)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(IncludeFixerContext::QuerySymbolInfo)

namespace llvm {
namespace yaml {

template <> struct MappingTraits<tooling::Range> {
  struct NormalizedRange {
    NormalizedRange(const IO &) : Offset(0), Length(0) {}

    NormalizedRange(const IO &, const tooling::Range &R)
        : Offset(R.getOffset()), Length(R.getLength()) {}

    tooling::Range denormalize(const IO &) {
      return tooling::Range(Offset, Length);
    }

    unsigned Offset;
    unsigned Length;
  };
  static void mapping(IO &IO, tooling::Range &Info) {
    MappingNormalization<NormalizedRange, tooling::Range> Keys(IO, Info);
    IO.mapRequired("Offset", Keys->Offset);
    IO.mapRequired("Length", Keys->Length);
  }
};

template <> struct MappingTraits<IncludeFixerContext::HeaderInfo> {
  static void mapping(IO &io, IncludeFixerContext::HeaderInfo &Info) {
    io.mapRequired("Header", Info.Header);
    io.mapRequired("QualifiedName", Info.QualifiedName);
  }
};

template <> struct MappingTraits<IncludeFixerContext::QuerySymbolInfo> {
  static void mapping(IO &io, IncludeFixerContext::QuerySymbolInfo &Info) {
    io.mapRequired("RawIdentifier", Info.RawIdentifier);
    io.mapRequired("Range", Info.Range);
  }
};

template <> struct MappingTraits<IncludeFixerContext> {
  static void mapping(IO &IO, IncludeFixerContext &Context) {
    IO.mapRequired("QuerySymbolInfos", Context.QuerySymbolInfos);
    IO.mapRequired("HeaderInfos", Context.HeaderInfos);
    IO.mapRequired("FilePath", Context.FilePath);
  }
};
} // namespace yaml
} // namespace llvm

namespace {
cl::OptionCategory IncludeFixerCategory("Tool options");

enum DatabaseFormatTy {
  fixed, ///< Hard-coded mapping.
  yaml,  ///< Yaml database created by find-all-symbols.
};

cl::opt<DatabaseFormatTy> DatabaseFormat(
    "db", cl::desc("Specify input format"),
    cl::values(clEnumVal(fixed, "Hard-coded mapping"),
               clEnumVal(yaml, "Yaml database created by find-all-symbols"),
               clEnumValEnd),
    cl::init(yaml), cl::cat(IncludeFixerCategory));

cl::opt<std::string> Input("input",
                           cl::desc("String to initialize the database"),
                           cl::cat(IncludeFixerCategory));

cl::opt<bool>
    MinimizeIncludePaths("minimize-paths",
                         cl::desc("Whether to minimize added include paths"),
                         cl::init(true), cl::cat(IncludeFixerCategory));

cl::opt<bool> Quiet("q", cl::desc("Reduce terminal output"), cl::init(false),
                    cl::cat(IncludeFixerCategory));

cl::opt<bool>
    STDINMode("stdin",
              cl::desc("Override source file's content (in the overlaying\n"
                       "virtual file system) with input from <stdin> and run\n"
                       "the tool on the new content with the compilation\n"
                       "options of the source file. This mode is currently\n"
                       "used for editor integration."),
              cl::init(false), cl::cat(IncludeFixerCategory));

cl::opt<bool> OutputHeaders(
    "output-headers",
    cl::desc("Print the symbol being queried and all its relevant headers in\n"
             "JSON format to stdout:\n"
             "  {\n"
             "    \"FilePath\": \"/path/to/foo.cc\",\n"
             "    \"QuerySymbolInfos\": [\n"
             "       {\"RawIdentifier\": \"foo\",\n"
             "        \"Range\": {\"Offset\": 0, \"Length\": 3}}\n"
             "    ],\n"
             "    \"HeaderInfos\": [ {\"Header\": \"\\\"foo_a.h\\\"\",\n"
             "                      \"QualifiedName\": \"a::foo\"} ]\n"
             "  }"),
    cl::init(false), cl::cat(IncludeFixerCategory));

cl::opt<std::string> InsertHeader(
    "insert-header",
    cl::desc("Insert a specific header. This should run with STDIN mode.\n"
             "The result is written to stdout. It is currently used for\n"
             "editor integration. Support YAML/JSON format:\n"
             "  -insert-header=\"{\n"
             "     FilePath: \"/path/to/foo.cc\",\n"
             "     QuerySymbolInfos: [\n"
             "       {RawIdentifier: foo,\n"
             "        Range: {Offset: 0, Length: 3}}\n"
             "     ],\n"
             "     HeaderInfos: [ {Headers: \"\\\"foo_a.h\\\"\",\n"
             "                     QualifiedName: \"a::foo\"} ]}\""),
    cl::init(""), cl::cat(IncludeFixerCategory));

cl::opt<std::string>
    Style("style",
          cl::desc("Fallback style for reformatting after inserting new\n"
                   "headers if there is no clang-format config file found."),
          cl::init("llvm"), cl::cat(IncludeFixerCategory));

std::unique_ptr<include_fixer::SymbolIndexManager>
createSymbolIndexManager(StringRef FilePath) {
  auto SymbolIndexMgr = llvm::make_unique<include_fixer::SymbolIndexManager>();
  switch (DatabaseFormat) {
  case fixed: {
    // Parse input and fill the database with it.
    // <symbol>=<header><, header...>
    // Multiple symbols can be given, separated by semicolons.
    std::map<std::string, std::vector<std::string>> SymbolsMap;
    SmallVector<StringRef, 4> SemicolonSplits;
    StringRef(Input).split(SemicolonSplits, ";");
    std::vector<find_all_symbols::SymbolInfo> Symbols;
    for (StringRef Pair : SemicolonSplits) {
      auto Split = Pair.split('=');
      std::vector<std::string> Headers;
      SmallVector<StringRef, 4> CommaSplits;
      Split.second.split(CommaSplits, ",");
      for (size_t I = 0, E = CommaSplits.size(); I != E; ++I)
        Symbols.push_back(find_all_symbols::SymbolInfo(
            Split.first.trim(),
            find_all_symbols::SymbolInfo::SymbolKind::Unknown,
            CommaSplits[I].trim(), 1, {}, /*NumOccurrences=*/E - I));
    }
    SymbolIndexMgr->addSymbolIndex(
        llvm::make_unique<include_fixer::InMemorySymbolIndex>(Symbols));
    break;
  }
  case yaml: {
    llvm::ErrorOr<std::unique_ptr<include_fixer::YamlSymbolIndex>> DB(nullptr);
    if (!Input.empty()) {
      DB = include_fixer::YamlSymbolIndex::createFromFile(Input);
    } else {
      // If we don't have any input file, look in the directory of the first
      // file and its parents.
      SmallString<128> AbsolutePath(tooling::getAbsolutePath(FilePath));
      StringRef Directory = llvm::sys::path::parent_path(AbsolutePath);
      DB = include_fixer::YamlSymbolIndex::createFromDirectory(
          Directory, "find_all_symbols_db.yaml");
    }

    if (!DB) {
      llvm::errs() << "Couldn't find YAML db: " << DB.getError().message()
                   << '\n';
      return nullptr;
    }

    SymbolIndexMgr->addSymbolIndex(std::move(*DB));
    break;
  }
  }
  return SymbolIndexMgr;
}

void writeToJson(llvm::raw_ostream &OS, const IncludeFixerContext& Context) {
  OS << "{\n"
     << "  \"FilePath\": \""
     << llvm::yaml::escape(Context.getFilePath()) << "\",\n"
     << "  \"QuerySymbolInfos\": [\n";
  for (const auto &Info : Context.getQuerySymbolInfos()) {
    OS << "     {\"RawIdentifier\": \"" << Info.RawIdentifier << "\",\n";
    OS << "      \"Range\":{";
    OS << "\"Offset\":" << Info.Range.getOffset() << ",";
    OS << "\"Length\":" << Info.Range.getLength() << "}}";
    if (&Info != &Context.getQuerySymbolInfos().back())
      OS << ",\n";
  }
  OS << "\n  ],\n";
  OS << "  \"HeaderInfos\": [\n";
  const auto &HeaderInfos = Context.getHeaderInfos();
  for (const auto &Info : HeaderInfos) {
    OS << "     {\"Header\": \"" << llvm::yaml::escape(Info.Header) << "\",\n"
       << "      \"QualifiedName\": \"" << Info.QualifiedName << "\"}";
    if (&Info != &HeaderInfos.back())
      OS << ",\n";
  }
  OS << "\n";
  OS << "  ]\n";
  OS << "}\n";
}

int includeFixerMain(int argc, const char **argv) {
  tooling::CommonOptionsParser options(argc, argv, IncludeFixerCategory);
  tooling::ClangTool tool(options.getCompilations(),
                          options.getSourcePathList());

  // In STDINMode, we override the file content with the <stdin> input.
  // Since `tool.mapVirtualFile` takes `StringRef`, we define `Code` outside of
  // the if-block so that `Code` is not released after the if-block.
  std::unique_ptr<llvm::MemoryBuffer> Code;
  if (STDINMode) {
    assert(options.getSourcePathList().size() == 1 &&
           "Expect exactly one file path in STDINMode.");
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> CodeOrErr =
        MemoryBuffer::getSTDIN();
    if (std::error_code EC = CodeOrErr.getError()) {
      errs() << EC.message() << "\n";
      return 1;
    }
    Code = std::move(CodeOrErr.get());
    if (Code->getBufferSize() == 0)
      return 0;  // Skip empty files.

    tool.mapVirtualFile(options.getSourcePathList().front(), Code->getBuffer());
  }

  if (!InsertHeader.empty()) {
    if (!STDINMode) {
      errs() << "Should be running in STDIN mode\n";
      return 1;
    }

    llvm::yaml::Input yin(InsertHeader);
    IncludeFixerContext Context;
    yin >> Context;

    const auto &HeaderInfos = Context.getHeaderInfos();
    assert(!HeaderInfos.empty());
    // We only accept one unique header.
    // Check all elements in HeaderInfos have the same header.
    bool IsUniqueHeader = std::equal(
        HeaderInfos.begin()+1, HeaderInfos.end(), HeaderInfos.begin(),
        [](const IncludeFixerContext::HeaderInfo &LHS,
           const IncludeFixerContext::HeaderInfo &RHS) {
          return LHS.Header == RHS.Header;
        });
    if (!IsUniqueHeader) {
      errs() << "Expect exactly one unique header.\n";
      return 1;
    }

    // If a header has multiple symbols, we won't add the missing namespace
    // qualifiers because we don't know which one is exactly used.
    //
    // Check whether all elements in HeaderInfos have the same qualified name.
    bool IsUniqueQualifiedName = std::equal(
        HeaderInfos.begin() + 1, HeaderInfos.end(), HeaderInfos.begin(),
        [](const IncludeFixerContext::HeaderInfo &LHS,
           const IncludeFixerContext::HeaderInfo &RHS) {
          return LHS.QualifiedName == RHS.QualifiedName;
        });
    format::FormatStyle InsertStyle =
        format::getStyle("file", Context.getFilePath(), Style);
    auto Replacements = clang::include_fixer::createIncludeFixerReplacements(
        Code->getBuffer(), Context, InsertStyle,
        /*AddQualifiers=*/IsUniqueQualifiedName);
    if (!Replacements) {
      errs() << "Failed to create replacements: "
             << llvm::toString(Replacements.takeError()) << "\n";
      return 1;
    }

    auto ChangedCode =
        tooling::applyAllReplacements(Code->getBuffer(), *Replacements);
    if (!ChangedCode) {
      llvm::errs() << llvm::toString(ChangedCode.takeError()) << "\n";
      return 1;
    }
    llvm::outs() << *ChangedCode;
    return 0;
  }

  // Set up data source.
  std::unique_ptr<include_fixer::SymbolIndexManager> SymbolIndexMgr =
      createSymbolIndexManager(options.getSourcePathList().front());
  if (!SymbolIndexMgr)
    return 1;

  // Now run our tool.
  std::vector<include_fixer::IncludeFixerContext> Contexts;
  include_fixer::IncludeFixerActionFactory Factory(*SymbolIndexMgr, Contexts,
                                                   Style, MinimizeIncludePaths);

  if (tool.run(&Factory) != 0) {
    llvm::errs()
        << "Clang died with a fatal error! (incorrect include paths?)\n";
    return 1;
  }

  assert(!Contexts.empty());

  if (OutputHeaders) {
    // FIXME: Print contexts of all processing files instead of the first one.
    writeToJson(llvm::outs(), Contexts.front());
    return 0;
  }

  std::vector<tooling::Replacements> FixerReplacements;
  for (const auto &Context : Contexts) {
    StringRef FilePath = Context.getFilePath();
    format::FormatStyle InsertStyle = format::getStyle("file", FilePath, Style);
    auto Buffer = llvm::MemoryBuffer::getFile(FilePath);
    if (!Buffer) {
      errs() << "Couldn't open file: " + FilePath.str() + ": "
             << Buffer.getError().message() + "\n";
      return 1;
    }

    auto Replacements = clang::include_fixer::createIncludeFixerReplacements(
        Buffer.get()->getBuffer(), Context, InsertStyle);
    if (!Replacements) {
      errs() << "Failed to create replacement: "
             << llvm::toString(Replacements.takeError()) << "\n";
      return 1;
    }
    FixerReplacements.push_back(*Replacements);
  }

  if (!Quiet) {
    for (const auto &Context : Contexts) {
      if (!Context.getHeaderInfos().empty()) {
        llvm::errs() << "Added #include "
                     << Context.getHeaderInfos().front().Header << " for "
                     << Context.getFilePath() << "\n";
      }
    }
  }

  if (STDINMode) {
    assert(FixerReplacements.size() == 1);
    auto ChangedCode = tooling::applyAllReplacements(Code->getBuffer(),
                                                     FixerReplacements.front());
    if (!ChangedCode) {
      llvm::errs() << llvm::toString(ChangedCode.takeError()) << "\n";
      return 1;
    }
    llvm::outs() << *ChangedCode;
    return 0;
  }

  // Set up a new source manager for applying the resulting replacements.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts(new DiagnosticOptions);
  DiagnosticsEngine Diagnostics(new DiagnosticIDs, &*DiagOpts);
  TextDiagnosticPrinter DiagnosticPrinter(outs(), &*DiagOpts);
  SourceManager SM(Diagnostics, tool.getFiles());
  Diagnostics.setClient(&DiagnosticPrinter, false);

  // Write replacements to disk.
  Rewriter Rewrites(SM, LangOptions());
  for (const auto Replacement : FixerReplacements) {
    if (!tooling::applyAllReplacements(Replacement, Rewrites)) {
      llvm::errs() << "Failed to apply replacements.\n";
      return 1;
    }
  }
  return Rewrites.overwriteChangedFiles();
}

} // namespace

int main(int argc, const char **argv) {
  return includeFixerMain(argc, argv);
}
