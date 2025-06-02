//===- ModuleMapFile.cpp - ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file handles parsing of modulemap files into a simple AST.
///
//===----------------------------------------------------------------------===//

#include "clang/Lex/ModuleMapFile.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/ModuleMap.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

using namespace clang;
using namespace modulemap;

namespace {
struct MMToken {
  enum TokenKind {
    Comma,
    ConfigMacros,
    Conflict,
    EndOfFile,
    HeaderKeyword,
    Identifier,
    Exclaim,
    ExcludeKeyword,
    ExplicitKeyword,
    ExportKeyword,
    ExportAsKeyword,
    ExternKeyword,
    FrameworkKeyword,
    LinkKeyword,
    ModuleKeyword,
    Period,
    PrivateKeyword,
    UmbrellaKeyword,
    UseKeyword,
    RequiresKeyword,
    Star,
    StringLiteral,
    IntegerLiteral,
    TextualKeyword,
    LBrace,
    RBrace,
    LSquare,
    RSquare
  } Kind;

  SourceLocation::UIntTy Location;
  unsigned StringLength;
  union {
    // If Kind != IntegerLiteral.
    const char *StringData;

    // If Kind == IntegerLiteral.
    uint64_t IntegerValue;
  };

  void clear() {
    Kind = EndOfFile;
    Location = 0;
    StringLength = 0;
    StringData = nullptr;
  }

  bool is(TokenKind K) const { return Kind == K; }

  SourceLocation getLocation() const {
    return SourceLocation::getFromRawEncoding(Location);
  }

  uint64_t getInteger() const {
    return Kind == IntegerLiteral ? IntegerValue : 0;
  }

  StringRef getString() const {
    return Kind == IntegerLiteral ? StringRef()
                                  : StringRef(StringData, StringLength);
  }
};

struct ModuleMapFileParser {
  // External context
  Lexer &L;
  DiagnosticsEngine &Diags;

  /// Parsed representation of the module map file
  ModuleMapFile MMF{};

  bool HadError = false;

  /// The current token.
  MMToken Tok{};

  bool parseTopLevelDecls();
  std::optional<ModuleDecl> parseModuleDecl(bool TopLevel);
  std::optional<ExternModuleDecl> parseExternModuleDecl();
  std::optional<ConfigMacrosDecl> parseConfigMacrosDecl();
  std::optional<ConflictDecl> parseConflictDecl();
  std::optional<ExportDecl> parseExportDecl();
  std::optional<ExportAsDecl> parseExportAsDecl();
  std::optional<UseDecl> parseUseDecl();
  std::optional<RequiresDecl> parseRequiresDecl();
  std::optional<HeaderDecl> parseHeaderDecl(MMToken::TokenKind LeadingToken,
                                            SourceLocation LeadingLoc);
  std::optional<ExcludeDecl> parseExcludeDecl(clang::SourceLocation LeadingLoc);
  std::optional<UmbrellaDirDecl>
  parseUmbrellaDirDecl(SourceLocation UmbrellaLoc);
  std::optional<LinkDecl> parseLinkDecl();

  SourceLocation consumeToken();
  void skipUntil(MMToken::TokenKind K);
  bool parseModuleId(ModuleId &Id);
  bool parseOptionalAttributes(ModuleAttributes &Attrs);

  SourceLocation getLocation() const { return Tok.getLocation(); };
};

std::string formatModuleId(const ModuleId &Id) {
  std::string result;
  {
    llvm::raw_string_ostream OS(result);

    for (unsigned I = 0, N = Id.size(); I != N; ++I) {
      if (I)
        OS << ".";
      OS << Id[I].first;
    }
  }

  return result;
}
} // end anonymous namespace

std::optional<ModuleMapFile>
modulemap::parseModuleMap(FileID ID, clang::DirectoryEntryRef Dir,
                          SourceManager &SM, DiagnosticsEngine &Diags,
                          bool IsSystem, unsigned *Offset) {
  std::optional<llvm::MemoryBufferRef> Buffer = SM.getBufferOrNone(ID);
  LangOptions LOpts;
  LOpts.LangStd = clang::LangStandard::lang_c99;
  Lexer L(SM.getLocForStartOfFile(ID), LOpts, Buffer->getBufferStart(),
          Buffer->getBufferStart() + (Offset ? *Offset : 0),
          Buffer->getBufferEnd());
  SourceLocation Start = L.getSourceLocation();

  ModuleMapFileParser Parser{L, Diags};
  bool Failed = Parser.parseTopLevelDecls();

  if (Offset) {
    auto Loc = SM.getDecomposedLoc(Parser.getLocation());
    assert(Loc.first == ID && "stopped in a different file?");
    *Offset = Loc.second;
  }

  if (Failed)
    return std::nullopt;
  Parser.MMF.ID = ID;
  Parser.MMF.Dir = Dir;
  Parser.MMF.Start = Start;
  Parser.MMF.IsSystem = IsSystem;
  return std::move(Parser.MMF);
}

bool ModuleMapFileParser::parseTopLevelDecls() {
  Tok.clear();
  consumeToken();
  do {
    switch (Tok.Kind) {
    case MMToken::EndOfFile:
      return HadError;
    case MMToken::ExternKeyword: {
      std::optional<ExternModuleDecl> EMD = parseExternModuleDecl();
      if (EMD)
        MMF.Decls.push_back(std::move(*EMD));
      break;
    }
    case MMToken::ExplicitKeyword:
    case MMToken::ModuleKeyword:
    case MMToken::FrameworkKeyword: {
      std::optional<ModuleDecl> MD = parseModuleDecl(true);
      if (MD)
        MMF.Decls.push_back(std::move(*MD));
      break;
    }
    case MMToken::Comma:
    case MMToken::ConfigMacros:
    case MMToken::Conflict:
    case MMToken::Exclaim:
    case MMToken::ExcludeKeyword:
    case MMToken::ExportKeyword:
    case MMToken::ExportAsKeyword:
    case MMToken::HeaderKeyword:
    case MMToken::Identifier:
    case MMToken::LBrace:
    case MMToken::LinkKeyword:
    case MMToken::LSquare:
    case MMToken::Period:
    case MMToken::PrivateKeyword:
    case MMToken::RBrace:
    case MMToken::RSquare:
    case MMToken::RequiresKeyword:
    case MMToken::Star:
    case MMToken::StringLiteral:
    case MMToken::IntegerLiteral:
    case MMToken::TextualKeyword:
    case MMToken::UmbrellaKeyword:
    case MMToken::UseKeyword:
      Diags.Report(Tok.getLocation(), diag::err_mmap_expected_module);
      HadError = true;
      consumeToken();
      break;
    }
  } while (true);
}

/// Parse a module declaration.
///
///   module-declaration:
///     'extern' 'module' module-id string-literal
///     'explicit'[opt] 'framework'[opt] 'module' module-id attributes[opt]
///       { module-member* }
///
///   module-member:
///     requires-declaration
///     header-declaration
///     submodule-declaration
///     export-declaration
///     export-as-declaration
///     link-declaration
///
///   submodule-declaration:
///     module-declaration
///     inferred-submodule-declaration
std::optional<ModuleDecl> ModuleMapFileParser::parseModuleDecl(bool TopLevel) {
  assert(Tok.is(MMToken::ExplicitKeyword) || Tok.is(MMToken::ModuleKeyword) ||
         Tok.is(MMToken::FrameworkKeyword));

  ModuleDecl MDecl;

  SourceLocation ExplicitLoc;
  MDecl.Explicit = false;
  MDecl.Framework = false;

  // Parse 'explicit' keyword, if present.
  if (Tok.is(MMToken::ExplicitKeyword)) {
    MDecl.Location = ExplicitLoc = consumeToken();
    MDecl.Explicit = true;
  }

  // Parse 'framework' keyword, if present.
  if (Tok.is(MMToken::FrameworkKeyword)) {
    SourceLocation FrameworkLoc = consumeToken();
    if (!MDecl.Location.isValid())
      MDecl.Location = FrameworkLoc;
    MDecl.Framework = true;
  }

  // Parse 'module' keyword.
  if (!Tok.is(MMToken::ModuleKeyword)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_module);
    consumeToken();
    HadError = true;
    return std::nullopt;
  }
  SourceLocation ModuleLoc = consumeToken();
  if (!MDecl.Location.isValid())
    MDecl.Location = ModuleLoc; // 'module' keyword

  // If we have a wildcard for the module name, this is an inferred submodule.
  // We treat it as a normal module at this point.
  if (Tok.is(MMToken::Star)) {
    SourceLocation StarLoc = consumeToken();
    MDecl.Id.push_back({"*", StarLoc});
    if (TopLevel && !MDecl.Framework) {
      Diags.Report(StarLoc, diag::err_mmap_top_level_inferred_submodule);
      HadError = true;
      return std::nullopt;
    }
  } else {
    // Parse the module name.
    if (parseModuleId(MDecl.Id)) {
      HadError = true;
      return std::nullopt;
    }
    if (!TopLevel) {
      if (MDecl.Id.size() > 1) {
        Diags.Report(MDecl.Id.front().second,
                     diag::err_mmap_nested_submodule_id)
            << SourceRange(MDecl.Id.front().second, MDecl.Id.back().second);

        HadError = true;
      }
    } else if (MDecl.Id.size() == 1 && MDecl.Explicit) {
      // Top-level modules can't be explicit.
      Diags.Report(ExplicitLoc, diag::err_mmap_explicit_top_level);
      MDecl.Explicit = false;
      HadError = true;
    }
  }

  // Parse the optional attribute list.
  if (parseOptionalAttributes(MDecl.Attrs))
    return std::nullopt;

  // Parse the opening brace.
  if (!Tok.is(MMToken::LBrace)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_lbrace)
        << MDecl.Id.back().first;
    HadError = true;
    return std::nullopt;
  }
  SourceLocation LBraceLoc = consumeToken();

  bool Done = false;
  do {
    std::optional<Decl> SubDecl;
    switch (Tok.Kind) {
    case MMToken::EndOfFile:
    case MMToken::RBrace:
      Done = true;
      break;

    case MMToken::ConfigMacros:
      // Only top-level modules can have configuration macros.
      if (!TopLevel)
        Diags.Report(Tok.getLocation(), diag::err_mmap_config_macro_submodule);
      SubDecl = parseConfigMacrosDecl();
      break;

    case MMToken::Conflict:
      SubDecl = parseConflictDecl();
      break;

    case MMToken::ExternKeyword:
      SubDecl = parseExternModuleDecl();
      break;

    case MMToken::ExplicitKeyword:
    case MMToken::FrameworkKeyword:
    case MMToken::ModuleKeyword:
      SubDecl = parseModuleDecl(false);
      break;

    case MMToken::ExportKeyword:
      SubDecl = parseExportDecl();
      break;

    case MMToken::ExportAsKeyword:
      if (!TopLevel) {
        Diags.Report(Tok.getLocation(), diag::err_mmap_submodule_export_as);
        parseExportAsDecl();
      } else
        SubDecl = parseExportAsDecl();
      break;

    case MMToken::UseKeyword:
      SubDecl = parseUseDecl();
      break;

    case MMToken::RequiresKeyword:
      SubDecl = parseRequiresDecl();
      break;

    case MMToken::TextualKeyword:
      SubDecl = parseHeaderDecl(MMToken::TextualKeyword, consumeToken());
      break;

    case MMToken::UmbrellaKeyword: {
      SourceLocation UmbrellaLoc = consumeToken();
      if (Tok.is(MMToken::HeaderKeyword))
        SubDecl = parseHeaderDecl(MMToken::UmbrellaKeyword, UmbrellaLoc);
      else
        SubDecl = parseUmbrellaDirDecl(UmbrellaLoc);
      break;
    }

    case MMToken::ExcludeKeyword: {
      SourceLocation ExcludeLoc = consumeToken();
      if (Tok.is(MMToken::HeaderKeyword))
        SubDecl = parseHeaderDecl(MMToken::ExcludeKeyword, ExcludeLoc);
      else
        SubDecl = parseExcludeDecl(ExcludeLoc);
      break;
    }

    case MMToken::PrivateKeyword:
      SubDecl = parseHeaderDecl(MMToken::PrivateKeyword, consumeToken());
      break;

    case MMToken::HeaderKeyword:
      SubDecl = parseHeaderDecl(MMToken::HeaderKeyword, consumeToken());
      break;

    case MMToken::LinkKeyword:
      SubDecl = parseLinkDecl();
      break;

    default:
      Diags.Report(Tok.getLocation(), diag::err_mmap_expected_member);
      consumeToken();
      break;
    }
    if (SubDecl)
      MDecl.Decls.push_back(std::move(*SubDecl));
  } while (!Done);

  if (Tok.is(MMToken::RBrace))
    consumeToken();
  else {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_rbrace);
    Diags.Report(LBraceLoc, diag::note_mmap_lbrace_match);
    HadError = true;
  }
  return std::move(MDecl);
}

std::optional<ExternModuleDecl> ModuleMapFileParser::parseExternModuleDecl() {
  assert(Tok.is(MMToken::ExternKeyword));
  ExternModuleDecl EMD;
  EMD.Location = consumeToken(); // 'extern' keyword

  // Parse 'module' keyword.
  if (!Tok.is(MMToken::ModuleKeyword)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_module);
    consumeToken();
    HadError = true;
    return std::nullopt;
  }
  consumeToken(); // 'module' keyword

  // Parse the module name.
  if (parseModuleId(EMD.Id)) {
    HadError = true;
    return std::nullopt;
  }

  // Parse the referenced module map file name.
  if (!Tok.is(MMToken::StringLiteral)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_mmap_file);
    HadError = true;
    return std::nullopt;
  }
  EMD.Path = Tok.getString();
  consumeToken(); // filename

  return std::move(EMD);
}

/// Parse a configuration macro declaration.
///
///   module-declaration:
///     'config_macros' attributes[opt] config-macro-list?
///
///   config-macro-list:
///     identifier (',' identifier)?
std::optional<ConfigMacrosDecl> ModuleMapFileParser::parseConfigMacrosDecl() {
  assert(Tok.is(MMToken::ConfigMacros));
  ConfigMacrosDecl CMDecl;
  CMDecl.Location = consumeToken();

  // Parse the optional attributes.
  ModuleAttributes Attrs;
  if (parseOptionalAttributes(Attrs))
    return std::nullopt;

  CMDecl.Exhaustive = Attrs.IsExhaustive;

  // If we don't have an identifier, we're done.
  // FIXME: Support macros with the same name as a keyword here.
  if (!Tok.is(MMToken::Identifier))
    return std::nullopt;

  // Consume the first identifier.
  CMDecl.Macros.push_back(Tok.getString());
  consumeToken();

  do {
    // If there's a comma, consume it.
    if (!Tok.is(MMToken::Comma))
      break;
    consumeToken();

    // We expect to see a macro name here.
    // FIXME: Support macros with the same name as a keyword here.
    if (!Tok.is(MMToken::Identifier)) {
      Diags.Report(Tok.getLocation(), diag::err_mmap_expected_config_macro);
      return std::nullopt;
    }

    // Consume the macro name.
    CMDecl.Macros.push_back(Tok.getString());
    consumeToken();
  } while (true);
  return std::move(CMDecl);
}

/// Parse a conflict declaration.
///
///   module-declaration:
///     'conflict' module-id ',' string-literal
std::optional<ConflictDecl> ModuleMapFileParser::parseConflictDecl() {
  assert(Tok.is(MMToken::Conflict));
  ConflictDecl CD;
  CD.Location = consumeToken();

  // Parse the module-id.
  if (parseModuleId(CD.Id))
    return std::nullopt;

  // Parse the ','.
  if (!Tok.is(MMToken::Comma)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_conflicts_comma)
        << SourceRange(CD.Location);
    return std::nullopt;
  }
  consumeToken();

  // Parse the message.
  if (!Tok.is(MMToken::StringLiteral)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_conflicts_message)
        << formatModuleId(CD.Id);
    return std::nullopt;
  }
  CD.Message = Tok.getString();
  consumeToken();
  return std::move(CD);
}

/// Parse a module export declaration.
///
///   export-declaration:
///     'export' wildcard-module-id
///
///   wildcard-module-id:
///     identifier
///     '*'
///     identifier '.' wildcard-module-id
std::optional<ExportDecl> ModuleMapFileParser::parseExportDecl() {
  assert(Tok.is(MMToken::ExportKeyword));
  ExportDecl ED;
  ED.Location = consumeToken();

  // Parse the module-id with an optional wildcard at the end.
  ED.Wildcard = false;
  do {
    // FIXME: Support string-literal module names here.
    if (Tok.is(MMToken::Identifier)) {
      ED.Id.push_back(
          std::make_pair(std::string(Tok.getString()), Tok.getLocation()));
      consumeToken();

      if (Tok.is(MMToken::Period)) {
        consumeToken();
        continue;
      }

      break;
    }

    if (Tok.is(MMToken::Star)) {
      ED.Wildcard = true;
      consumeToken();
      break;
    }

    Diags.Report(Tok.getLocation(), diag::err_mmap_module_id);
    HadError = true;
    return std::nullopt;
  } while (true);

  return std::move(ED);
}

/// Parse a module export_as declaration.
///
///   export-as-declaration:
///     'export_as' identifier
std::optional<ExportAsDecl> ModuleMapFileParser::parseExportAsDecl() {
  assert(Tok.is(MMToken::ExportAsKeyword));
  ExportAsDecl EAD;
  EAD.Location = consumeToken();

  if (!Tok.is(MMToken::Identifier)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_module_id);
    HadError = true;
    return std::nullopt;
  }

  if (parseModuleId(EAD.Id))
    return std::nullopt;
  if (EAD.Id.size() > 1)
    Diags.Report(EAD.Id[1].second, diag::err_mmap_qualified_export_as);
  return std::move(EAD);
}

/// Parse a module use declaration.
///
///   use-declaration:
///     'use' wildcard-module-id
std::optional<UseDecl> ModuleMapFileParser::parseUseDecl() {
  assert(Tok.is(MMToken::UseKeyword));
  UseDecl UD;
  UD.Location = consumeToken();
  if (parseModuleId(UD.Id))
    return std::nullopt;
  return std::move(UD);
}

/// Parse a requires declaration.
///
///   requires-declaration:
///     'requires' feature-list
///
///   feature-list:
///     feature ',' feature-list
///     feature
///
///   feature:
///     '!'[opt] identifier
std::optional<RequiresDecl> ModuleMapFileParser::parseRequiresDecl() {
  assert(Tok.is(MMToken::RequiresKeyword));
  RequiresDecl RD;
  RD.Location = consumeToken();

  // Parse the feature-list.
  do {
    bool RequiredState = true;
    if (Tok.is(MMToken::Exclaim)) {
      RequiredState = false;
      consumeToken();
    }

    if (!Tok.is(MMToken::Identifier)) {
      Diags.Report(Tok.getLocation(), diag::err_mmap_expected_feature);
      HadError = true;
      return std::nullopt;
    }

    // Consume the feature name.
    RequiresFeature RF;
    RF.Feature = Tok.getString();
    RF.Location = consumeToken();
    RF.RequiredState = RequiredState;

    RD.Features.push_back(std::move(RF));

    if (!Tok.is(MMToken::Comma))
      break;

    // Consume the comma.
    consumeToken();
  } while (true);
  return std::move(RD);
}

/// Parse a header declaration.
///
///   header-declaration:
///     'textual'[opt] 'header' string-literal
///     'private' 'textual'[opt] 'header' string-literal
///     'exclude' 'header' string-literal
///     'umbrella' 'header' string-literal
std::optional<HeaderDecl>
ModuleMapFileParser::parseHeaderDecl(MMToken::TokenKind LeadingToken,
                                     clang::SourceLocation LeadingLoc) {
  HeaderDecl HD;
  HD.Private = false;
  HD.Excluded = false;
  HD.Textual = false;
  // We've already consumed the first token.
  HD.Location = LeadingLoc;

  if (LeadingToken == MMToken::PrivateKeyword) {
    HD.Private = true;
    // 'private' may optionally be followed by 'textual'.
    if (Tok.is(MMToken::TextualKeyword)) {
      HD.Textual = true;
      LeadingToken = Tok.Kind;
      consumeToken();
    }
  } else if (LeadingToken == MMToken::ExcludeKeyword)
    HD.Excluded = true;
  else if (LeadingToken == MMToken::TextualKeyword)
    HD.Textual = true;

  if (LeadingToken != MMToken::HeaderKeyword) {
    if (!Tok.is(MMToken::HeaderKeyword)) {
      Diags.Report(Tok.getLocation(), diag::err_mmap_expected_header)
          << (LeadingToken == MMToken::PrivateKeyword   ? "private"
              : LeadingToken == MMToken::ExcludeKeyword ? "exclude"
              : LeadingToken == MMToken::TextualKeyword ? "textual"
                                                        : "umbrella");
      return std::nullopt;
    }
    consumeToken();
  }

  // Parse the header name.
  if (!Tok.is(MMToken::StringLiteral)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_header) << "header";
    HadError = true;
    return std::nullopt;
  }
  HD.Path = Tok.getString();
  HD.PathLoc = consumeToken();
  HD.Umbrella = LeadingToken == MMToken::UmbrellaKeyword;

  // If we were given stat information, parse it so we can skip looking for
  // the file.
  if (Tok.is(MMToken::LBrace)) {
    SourceLocation LBraceLoc = consumeToken();

    while (!Tok.is(MMToken::RBrace) && !Tok.is(MMToken::EndOfFile)) {
      enum Attribute { Size, ModTime, Unknown };
      StringRef Str = Tok.getString();
      SourceLocation Loc = consumeToken();
      switch (llvm::StringSwitch<Attribute>(Str)
                  .Case("size", Size)
                  .Case("mtime", ModTime)
                  .Default(Unknown)) {
      case Size:
        if (HD.Size)
          Diags.Report(Loc, diag::err_mmap_duplicate_header_attribute) << Str;
        if (!Tok.is(MMToken::IntegerLiteral)) {
          Diags.Report(Tok.getLocation(),
                       diag::err_mmap_invalid_header_attribute_value)
              << Str;
          skipUntil(MMToken::RBrace);
          break;
        }
        HD.Size = Tok.getInteger();
        consumeToken();
        break;

      case ModTime:
        if (HD.MTime)
          Diags.Report(Loc, diag::err_mmap_duplicate_header_attribute) << Str;
        if (!Tok.is(MMToken::IntegerLiteral)) {
          Diags.Report(Tok.getLocation(),
                       diag::err_mmap_invalid_header_attribute_value)
              << Str;
          skipUntil(MMToken::RBrace);
          break;
        }
        HD.MTime = Tok.getInteger();
        consumeToken();
        break;

      case Unknown:
        Diags.Report(Loc, diag::err_mmap_expected_header_attribute);
        skipUntil(MMToken::RBrace);
        break;
      }
    }

    if (Tok.is(MMToken::RBrace))
      consumeToken();
    else {
      Diags.Report(Tok.getLocation(), diag::err_mmap_expected_rbrace);
      Diags.Report(LBraceLoc, diag::note_mmap_lbrace_match);
      HadError = true;
    }
  }
  return std::move(HD);
}

/// Parse an exclude declaration.
///
/// exclude-declaration:
///   'exclude' identifier
std::optional<ExcludeDecl>
ModuleMapFileParser::parseExcludeDecl(clang::SourceLocation LeadingLoc) {
  // FIXME: Support string-literal module names here.
  if (!Tok.is(MMToken::Identifier)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_missing_exclude_name);
    HadError = true;
    return std::nullopt;
  }

  ExcludeDecl ED;
  ED.Location = LeadingLoc;
  ED.Module = Tok.getString();
  consumeToken();
  return std::move(ED);
}

/// Parse an umbrella directory declaration.
///
///   umbrella-dir-declaration:
///     umbrella string-literal
std::optional<UmbrellaDirDecl>
ModuleMapFileParser::parseUmbrellaDirDecl(clang::SourceLocation UmbrellaLoc) {
  UmbrellaDirDecl UDD;
  UDD.Location = UmbrellaLoc;
  // Parse the directory name.
  if (!Tok.is(MMToken::StringLiteral)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_header)
        << "umbrella";
    HadError = true;
    return std::nullopt;
  }

  UDD.Path = Tok.getString();
  consumeToken();
  return std::move(UDD);
}

/// Parse a link declaration.
///
///   module-declaration:
///     'link' 'framework'[opt] string-literal
std::optional<LinkDecl> ModuleMapFileParser::parseLinkDecl() {
  assert(Tok.is(MMToken::LinkKeyword));
  LinkDecl LD;
  LD.Location = consumeToken();

  // Parse the optional 'framework' keyword.
  LD.Framework = false;
  if (Tok.is(MMToken::FrameworkKeyword)) {
    consumeToken();
    LD.Framework = true;
  }

  // Parse the library name
  if (!Tok.is(MMToken::StringLiteral)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_library_name)
        << LD.Framework << SourceRange(LD.Location);
    HadError = true;
    return std::nullopt;
  }

  LD.Library = Tok.getString();
  consumeToken();
  return std::move(LD);
}

SourceLocation ModuleMapFileParser::consumeToken() {
  SourceLocation Result = Tok.getLocation();

retry:
  Tok.clear();
  Token LToken;
  L.LexFromRawLexer(LToken);
  Tok.Location = LToken.getLocation().getRawEncoding();
  switch (LToken.getKind()) {
  case tok::raw_identifier: {
    StringRef RI = LToken.getRawIdentifier();
    Tok.StringData = RI.data();
    Tok.StringLength = RI.size();
    Tok.Kind = llvm::StringSwitch<MMToken::TokenKind>(RI)
                   .Case("config_macros", MMToken::ConfigMacros)
                   .Case("conflict", MMToken::Conflict)
                   .Case("exclude", MMToken::ExcludeKeyword)
                   .Case("explicit", MMToken::ExplicitKeyword)
                   .Case("export", MMToken::ExportKeyword)
                   .Case("export_as", MMToken::ExportAsKeyword)
                   .Case("extern", MMToken::ExternKeyword)
                   .Case("framework", MMToken::FrameworkKeyword)
                   .Case("header", MMToken::HeaderKeyword)
                   .Case("link", MMToken::LinkKeyword)
                   .Case("module", MMToken::ModuleKeyword)
                   .Case("private", MMToken::PrivateKeyword)
                   .Case("requires", MMToken::RequiresKeyword)
                   .Case("textual", MMToken::TextualKeyword)
                   .Case("umbrella", MMToken::UmbrellaKeyword)
                   .Case("use", MMToken::UseKeyword)
                   .Default(MMToken::Identifier);
    break;
  }

  case tok::comma:
    Tok.Kind = MMToken::Comma;
    break;

  case tok::eof:
    Tok.Kind = MMToken::EndOfFile;
    break;

  case tok::l_brace:
    Tok.Kind = MMToken::LBrace;
    break;

  case tok::l_square:
    Tok.Kind = MMToken::LSquare;
    break;

  case tok::period:
    Tok.Kind = MMToken::Period;
    break;

  case tok::r_brace:
    Tok.Kind = MMToken::RBrace;
    break;

  case tok::r_square:
    Tok.Kind = MMToken::RSquare;
    break;

  case tok::star:
    Tok.Kind = MMToken::Star;
    break;

  case tok::exclaim:
    Tok.Kind = MMToken::Exclaim;
    break;

  case tok::string_literal: {
    if (LToken.hasUDSuffix()) {
      Diags.Report(LToken.getLocation(), diag::err_invalid_string_udl);
      HadError = true;
      goto retry;
    }

    // Form the token.
    Tok.Kind = MMToken::StringLiteral;
    Tok.StringData = LToken.getLiteralData() + 1;
    Tok.StringLength = LToken.getLength() - 2;
    break;
  }

  case tok::numeric_constant: {
    // We don't support any suffixes or other complications.
    uint64_t Value;
    if (StringRef(LToken.getLiteralData(), LToken.getLength())
            .getAsInteger(0, Value)) {
      Diags.Report(Tok.getLocation(), diag::err_mmap_unknown_token);
      HadError = true;
      goto retry;
    }

    Tok.Kind = MMToken::IntegerLiteral;
    Tok.IntegerValue = Value;
    break;
  }

  case tok::comment:
    goto retry;

  case tok::hash:
    // A module map can be terminated prematurely by
    //   #pragma clang module contents
    // When building the module, we'll treat the rest of the file as the
    // contents of the module.
    {
      auto NextIsIdent = [&](StringRef Str) -> bool {
        L.LexFromRawLexer(LToken);
        return !LToken.isAtStartOfLine() && LToken.is(tok::raw_identifier) &&
               LToken.getRawIdentifier() == Str;
      };
      if (NextIsIdent("pragma") && NextIsIdent("clang") &&
          NextIsIdent("module") && NextIsIdent("contents")) {
        Tok.Kind = MMToken::EndOfFile;
        break;
      }
    }
    [[fallthrough]];

  default:
    Diags.Report(Tok.getLocation(), diag::err_mmap_unknown_token);
    HadError = true;
    goto retry;
  }

  return Result;
}

void ModuleMapFileParser::skipUntil(MMToken::TokenKind K) {
  unsigned braceDepth = 0;
  unsigned squareDepth = 0;
  do {
    switch (Tok.Kind) {
    case MMToken::EndOfFile:
      return;

    case MMToken::LBrace:
      if (Tok.is(K) && braceDepth == 0 && squareDepth == 0)
        return;

      ++braceDepth;
      break;

    case MMToken::LSquare:
      if (Tok.is(K) && braceDepth == 0 && squareDepth == 0)
        return;

      ++squareDepth;
      break;

    case MMToken::RBrace:
      if (braceDepth > 0)
        --braceDepth;
      else if (Tok.is(K))
        return;
      break;

    case MMToken::RSquare:
      if (squareDepth > 0)
        --squareDepth;
      else if (Tok.is(K))
        return;
      break;

    default:
      if (braceDepth == 0 && squareDepth == 0 && Tok.is(K))
        return;
      break;
    }

    consumeToken();
  } while (true);
}

/// Parse a module-id.
///
///   module-id:
///     identifier
///     identifier '.' module-id
///
/// \returns true if an error occurred, false otherwise.
bool ModuleMapFileParser::parseModuleId(ModuleId &Id) {
  Id.clear();
  do {
    if (Tok.is(MMToken::Identifier) || Tok.is(MMToken::StringLiteral)) {
      Id.push_back(
          std::make_pair(std::string(Tok.getString()), Tok.getLocation()));
      consumeToken();
    } else {
      Diags.Report(Tok.getLocation(), diag::err_mmap_expected_module_name);
      return true;
    }

    if (!Tok.is(MMToken::Period))
      break;

    consumeToken();
  } while (true);

  return false;
}

/// Parse optional attributes.
///
///   attributes:
///     attribute attributes
///     attribute
///
///   attribute:
///     [ identifier ]
///
/// \param Attrs Will be filled in with the parsed attributes.
///
/// \returns true if an error occurred, false otherwise.
bool ModuleMapFileParser::parseOptionalAttributes(ModuleAttributes &Attrs) {
  bool Error = false;

  while (Tok.is(MMToken::LSquare)) {
    // Consume the '['.
    SourceLocation LSquareLoc = consumeToken();

    // Check whether we have an attribute name here.
    if (!Tok.is(MMToken::Identifier)) {
      Diags.Report(Tok.getLocation(), diag::err_mmap_expected_attribute);
      skipUntil(MMToken::RSquare);
      if (Tok.is(MMToken::RSquare))
        consumeToken();
      Error = true;
    }

    /// Enumerates the known attributes.
    enum AttributeKind {
      /// An unknown attribute.
      AT_unknown,

      /// The 'system' attribute.
      AT_system,

      /// The 'extern_c' attribute.
      AT_extern_c,

      /// The 'exhaustive' attribute.
      AT_exhaustive,

      /// The 'no_undeclared_includes' attribute.
      AT_no_undeclared_includes
    };

    // Decode the attribute name.
    AttributeKind Attribute =
        llvm::StringSwitch<AttributeKind>(Tok.getString())
            .Case("exhaustive", AT_exhaustive)
            .Case("extern_c", AT_extern_c)
            .Case("no_undeclared_includes", AT_no_undeclared_includes)
            .Case("system", AT_system)
            .Default(AT_unknown);
    switch (Attribute) {
    case AT_unknown:
      Diags.Report(Tok.getLocation(), diag::warn_mmap_unknown_attribute)
          << Tok.getString();
      break;

    case AT_system:
      Attrs.IsSystem = true;
      break;

    case AT_extern_c:
      Attrs.IsExternC = true;
      break;

    case AT_exhaustive:
      Attrs.IsExhaustive = true;
      break;

    case AT_no_undeclared_includes:
      Attrs.NoUndeclaredIncludes = true;
      break;
    }
    consumeToken();

    // Consume the ']'.
    if (!Tok.is(MMToken::RSquare)) {
      Diags.Report(Tok.getLocation(), diag::err_mmap_expected_rsquare);
      Diags.Report(LSquareLoc, diag::note_mmap_lsquare_match);
      skipUntil(MMToken::RSquare);
      Error = true;
    }

    if (Tok.is(MMToken::RSquare))
      consumeToken();
  }

  if (Error)
    HadError = true;

  return Error;
}

static void dumpModule(const ModuleDecl &MD, llvm::raw_ostream &out, int depth);

static void dumpExternModule(const ExternModuleDecl &EMD,
                             llvm::raw_ostream &out, int depth) {
  out.indent(depth * 2);
  out << "extern module " << formatModuleId(EMD.Id) << " \"" << EMD.Path
      << "\"\n";
}

static void dumpDecls(ArrayRef<Decl> Decls, llvm::raw_ostream &out, int depth) {
  for (const auto &Decl : Decls) {
    std::visit(llvm::makeVisitor(
                   [&](const RequiresDecl &RD) {
                     out.indent(depth * 2);
                     out << "requires\n";
                   },
                   [&](const HeaderDecl &HD) {
                     out.indent(depth * 2);
                     if (HD.Private)
                       out << "private ";
                     if (HD.Textual)
                       out << "textual ";
                     if (HD.Excluded)
                       out << "excluded ";
                     if (HD.Umbrella)
                       out << "umbrella ";
                     out << "header \"" << HD.Path << "\"\n";
                   },
                   [&](const UmbrellaDirDecl &UDD) {
                     out.indent(depth * 2);
                     out << "umbrella\n";
                   },
                   [&](const ModuleDecl &MD) { dumpModule(MD, out, depth); },
                   [&](const ExcludeDecl &ED) {
                     out.indent(depth * 2);
                     out << "exclude " << ED.Module << "\n";
                   },
                   [&](const ExportDecl &ED) {
                     out.indent(depth * 2);
                     out << "export "
                         << (ED.Wildcard ? "*" : formatModuleId(ED.Id)) << "\n";
                   },
                   [&](const ExportAsDecl &EAD) {
                     out.indent(depth * 2);
                     out << "export as\n";
                   },
                   [&](const ExternModuleDecl &EMD) {
                     dumpExternModule(EMD, out, depth);
                   },
                   [&](const UseDecl &UD) {
                     out.indent(depth * 2);
                     out << "use\n";
                   },
                   [&](const LinkDecl &LD) {
                     out.indent(depth * 2);
                     out << "link\n";
                   },
                   [&](const ConfigMacrosDecl &CMD) {
                     out.indent(depth * 2);
                     out << "config_macros ";
                     if (CMD.Exhaustive)
                       out << "[exhaustive] ";
                     for (auto Macro : CMD.Macros) {
                       out << Macro << " ";
                     }
                     out << "\n";
                   },
                   [&](const ConflictDecl &CD) {
                     out.indent(depth * 2);
                     out << "conflicts\n";
                   }),
               Decl);
  }
}

static void dumpModule(const ModuleDecl &MD, llvm::raw_ostream &out,
                       int depth) {
  out.indent(depth * 2);
  out << "module " << formatModuleId(MD.Id) << "\n";
  dumpDecls(MD.Decls, out, depth + 1);
}

void ModuleMapFile::dump(llvm::raw_ostream &out) const {
  for (const auto &Decl : Decls) {
    std::visit(
        llvm::makeVisitor([&](const ModuleDecl &MD) { dumpModule(MD, out, 0); },
                          [&](const ExternModuleDecl &EMD) {
                            dumpExternModule(EMD, out, 0);
                          }),
        Decl);
  }
}
