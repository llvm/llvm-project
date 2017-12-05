//===--- Format.cpp - Format C++ code -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements functions declared in Format.h. This will be
/// split into separate files as we go.
///
//===----------------------------------------------------------------------===//

#include "clang/Format/Format.h"
#include "AffectedRangeManager.h"
#include "ContinuationIndenter.h"
#include "FormatInternal.h"
#include "FormatTokenLexer.h"
#include "NamespaceEndCommentsFixer.h"
#include "SortJavaScriptImports.h"
#include "TokenAnalyzer.h"
#include "TokenAnnotator.h"
#include "UnwrappedLineFormatter.h"
#include "UnwrappedLineParser.h"
#include "UsingDeclarationsSorter.h"
#include "WhitespaceManager.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/VirtualFileSystem.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/YAMLTraits.h"
#include <algorithm>
#include <memory>
#include <string>

#define DEBUG_TYPE "format-formatter"

using clang::format::FormatStyle;

LLVM_YAML_IS_SEQUENCE_VECTOR(clang::format::FormatStyle::IncludeCategory)
LLVM_YAML_IS_SEQUENCE_VECTOR(clang::format::FormatStyle::RawStringFormat)

namespace llvm {
namespace yaml {
template <> struct ScalarEnumerationTraits<FormatStyle::LanguageKind> {
  static void enumeration(IO &IO, FormatStyle::LanguageKind &Value) {
    IO.enumCase(Value, "Cpp", FormatStyle::LK_Cpp);
    IO.enumCase(Value, "Java", FormatStyle::LK_Java);
    IO.enumCase(Value, "JavaScript", FormatStyle::LK_JavaScript);
    IO.enumCase(Value, "ObjC", FormatStyle::LK_ObjC);
    IO.enumCase(Value, "Proto", FormatStyle::LK_Proto);
    IO.enumCase(Value, "TableGen", FormatStyle::LK_TableGen);
    IO.enumCase(Value, "TextProto", FormatStyle::LK_TextProto);
  }
};

template <> struct ScalarEnumerationTraits<FormatStyle::LanguageStandard> {
  static void enumeration(IO &IO, FormatStyle::LanguageStandard &Value) {
    IO.enumCase(Value, "Cpp03", FormatStyle::LS_Cpp03);
    IO.enumCase(Value, "C++03", FormatStyle::LS_Cpp03);
    IO.enumCase(Value, "Cpp11", FormatStyle::LS_Cpp11);
    IO.enumCase(Value, "C++11", FormatStyle::LS_Cpp11);
    IO.enumCase(Value, "Auto", FormatStyle::LS_Auto);
  }
};

template <> struct ScalarEnumerationTraits<FormatStyle::UseTabStyle> {
  static void enumeration(IO &IO, FormatStyle::UseTabStyle &Value) {
    IO.enumCase(Value, "Never", FormatStyle::UT_Never);
    IO.enumCase(Value, "false", FormatStyle::UT_Never);
    IO.enumCase(Value, "Always", FormatStyle::UT_Always);
    IO.enumCase(Value, "true", FormatStyle::UT_Always);
    IO.enumCase(Value, "ForIndentation", FormatStyle::UT_ForIndentation);
    IO.enumCase(Value, "ForContinuationAndIndentation",
                FormatStyle::UT_ForContinuationAndIndentation);
  }
};

template <> struct ScalarEnumerationTraits<FormatStyle::JavaScriptQuoteStyle> {
  static void enumeration(IO &IO, FormatStyle::JavaScriptQuoteStyle &Value) {
    IO.enumCase(Value, "Leave", FormatStyle::JSQS_Leave);
    IO.enumCase(Value, "Single", FormatStyle::JSQS_Single);
    IO.enumCase(Value, "Double", FormatStyle::JSQS_Double);
  }
};

template <> struct ScalarEnumerationTraits<FormatStyle::ShortFunctionStyle> {
  static void enumeration(IO &IO, FormatStyle::ShortFunctionStyle &Value) {
    IO.enumCase(Value, "None", FormatStyle::SFS_None);
    IO.enumCase(Value, "false", FormatStyle::SFS_None);
    IO.enumCase(Value, "All", FormatStyle::SFS_All);
    IO.enumCase(Value, "true", FormatStyle::SFS_All);
    IO.enumCase(Value, "Inline", FormatStyle::SFS_Inline);
    IO.enumCase(Value, "InlineOnly", FormatStyle::SFS_InlineOnly);
    IO.enumCase(Value, "Empty", FormatStyle::SFS_Empty);
  }
};

template <> struct ScalarEnumerationTraits<FormatStyle::BinaryOperatorStyle> {
  static void enumeration(IO &IO, FormatStyle::BinaryOperatorStyle &Value) {
    IO.enumCase(Value, "All", FormatStyle::BOS_All);
    IO.enumCase(Value, "true", FormatStyle::BOS_All);
    IO.enumCase(Value, "None", FormatStyle::BOS_None);
    IO.enumCase(Value, "false", FormatStyle::BOS_None);
    IO.enumCase(Value, "NonAssignment", FormatStyle::BOS_NonAssignment);
  }
};

template <> struct ScalarEnumerationTraits<FormatStyle::BraceBreakingStyle> {
  static void enumeration(IO &IO, FormatStyle::BraceBreakingStyle &Value) {
    IO.enumCase(Value, "Attach", FormatStyle::BS_Attach);
    IO.enumCase(Value, "Linux", FormatStyle::BS_Linux);
    IO.enumCase(Value, "Mozilla", FormatStyle::BS_Mozilla);
    IO.enumCase(Value, "Stroustrup", FormatStyle::BS_Stroustrup);
    IO.enumCase(Value, "Allman", FormatStyle::BS_Allman);
    IO.enumCase(Value, "GNU", FormatStyle::BS_GNU);
    IO.enumCase(Value, "WebKit", FormatStyle::BS_WebKit);
    IO.enumCase(Value, "Custom", FormatStyle::BS_Custom);
  }
};

template <>
struct ScalarEnumerationTraits<FormatStyle::BreakConstructorInitializersStyle> {
  static void
  enumeration(IO &IO, FormatStyle::BreakConstructorInitializersStyle &Value) {
    IO.enumCase(Value, "BeforeColon", FormatStyle::BCIS_BeforeColon);
    IO.enumCase(Value, "BeforeComma", FormatStyle::BCIS_BeforeComma);
    IO.enumCase(Value, "AfterColon", FormatStyle::BCIS_AfterColon);
  }
};

template <>
struct ScalarEnumerationTraits<FormatStyle::PPDirectiveIndentStyle> {
  static void enumeration(IO &IO, FormatStyle::PPDirectiveIndentStyle &Value) {
    IO.enumCase(Value, "None", FormatStyle::PPDIS_None);
    IO.enumCase(Value, "AfterHash", FormatStyle::PPDIS_AfterHash);
  }
};

template <>
struct ScalarEnumerationTraits<FormatStyle::ReturnTypeBreakingStyle> {
  static void enumeration(IO &IO, FormatStyle::ReturnTypeBreakingStyle &Value) {
    IO.enumCase(Value, "None", FormatStyle::RTBS_None);
    IO.enumCase(Value, "All", FormatStyle::RTBS_All);
    IO.enumCase(Value, "TopLevel", FormatStyle::RTBS_TopLevel);
    IO.enumCase(Value, "TopLevelDefinitions",
                FormatStyle::RTBS_TopLevelDefinitions);
    IO.enumCase(Value, "AllDefinitions", FormatStyle::RTBS_AllDefinitions);
  }
};

template <>
struct ScalarEnumerationTraits<FormatStyle::DefinitionReturnTypeBreakingStyle> {
  static void
  enumeration(IO &IO, FormatStyle::DefinitionReturnTypeBreakingStyle &Value) {
    IO.enumCase(Value, "None", FormatStyle::DRTBS_None);
    IO.enumCase(Value, "All", FormatStyle::DRTBS_All);
    IO.enumCase(Value, "TopLevel", FormatStyle::DRTBS_TopLevel);

    // For backward compatibility.
    IO.enumCase(Value, "false", FormatStyle::DRTBS_None);
    IO.enumCase(Value, "true", FormatStyle::DRTBS_All);
  }
};

template <>
struct ScalarEnumerationTraits<FormatStyle::NamespaceIndentationKind> {
  static void enumeration(IO &IO,
                          FormatStyle::NamespaceIndentationKind &Value) {
    IO.enumCase(Value, "None", FormatStyle::NI_None);
    IO.enumCase(Value, "Inner", FormatStyle::NI_Inner);
    IO.enumCase(Value, "All", FormatStyle::NI_All);
  }
};

template <> struct ScalarEnumerationTraits<FormatStyle::BracketAlignmentStyle> {
  static void enumeration(IO &IO, FormatStyle::BracketAlignmentStyle &Value) {
    IO.enumCase(Value, "Align", FormatStyle::BAS_Align);
    IO.enumCase(Value, "DontAlign", FormatStyle::BAS_DontAlign);
    IO.enumCase(Value, "AlwaysBreak", FormatStyle::BAS_AlwaysBreak);

    // For backward compatibility.
    IO.enumCase(Value, "true", FormatStyle::BAS_Align);
    IO.enumCase(Value, "false", FormatStyle::BAS_DontAlign);
  }
};

template <>
struct ScalarEnumerationTraits<FormatStyle::EscapedNewlineAlignmentStyle> {
  static void enumeration(IO &IO,
                          FormatStyle::EscapedNewlineAlignmentStyle &Value) {
    IO.enumCase(Value, "DontAlign", FormatStyle::ENAS_DontAlign);
    IO.enumCase(Value, "Left", FormatStyle::ENAS_Left);
    IO.enumCase(Value, "Right", FormatStyle::ENAS_Right);

    // For backward compatibility.
    IO.enumCase(Value, "true", FormatStyle::ENAS_Left);
    IO.enumCase(Value, "false", FormatStyle::ENAS_Right);
  }
};

template <> struct ScalarEnumerationTraits<FormatStyle::PointerAlignmentStyle> {
  static void enumeration(IO &IO, FormatStyle::PointerAlignmentStyle &Value) {
    IO.enumCase(Value, "Middle", FormatStyle::PAS_Middle);
    IO.enumCase(Value, "Left", FormatStyle::PAS_Left);
    IO.enumCase(Value, "Right", FormatStyle::PAS_Right);

    // For backward compatibility.
    IO.enumCase(Value, "true", FormatStyle::PAS_Left);
    IO.enumCase(Value, "false", FormatStyle::PAS_Right);
  }
};

template <>
struct ScalarEnumerationTraits<FormatStyle::SpaceBeforeParensOptions> {
  static void enumeration(IO &IO,
                          FormatStyle::SpaceBeforeParensOptions &Value) {
    IO.enumCase(Value, "Never", FormatStyle::SBPO_Never);
    IO.enumCase(Value, "ControlStatements",
                FormatStyle::SBPO_ControlStatements);
    IO.enumCase(Value, "Always", FormatStyle::SBPO_Always);

    // For backward compatibility.
    IO.enumCase(Value, "false", FormatStyle::SBPO_Never);
    IO.enumCase(Value, "true", FormatStyle::SBPO_ControlStatements);
  }
};

template <> struct MappingTraits<FormatStyle> {
  static void mapping(IO &IO, FormatStyle &Style) {
    // When reading, read the language first, we need it for getPredefinedStyle.
    IO.mapOptional("Language", Style.Language);

    if (IO.outputting()) {
      StringRef StylesArray[] = {"LLVM",    "Google", "Chromium",
                                 "Mozilla", "WebKit", "GNU"};
      ArrayRef<StringRef> Styles(StylesArray);
      for (size_t i = 0, e = Styles.size(); i < e; ++i) {
        StringRef StyleName(Styles[i]);
        FormatStyle PredefinedStyle;
        if (getPredefinedStyle(StyleName, Style.Language, &PredefinedStyle) &&
            Style == PredefinedStyle) {
          IO.mapOptional("# BasedOnStyle", StyleName);
          break;
        }
      }
    } else {
      StringRef BasedOnStyle;
      IO.mapOptional("BasedOnStyle", BasedOnStyle);
      if (!BasedOnStyle.empty()) {
        FormatStyle::LanguageKind OldLanguage = Style.Language;
        FormatStyle::LanguageKind Language =
            ((FormatStyle *)IO.getContext())->Language;
        if (!getPredefinedStyle(BasedOnStyle, Language, &Style)) {
          IO.setError(Twine("Unknown value for BasedOnStyle: ", BasedOnStyle));
          return;
        }
        Style.Language = OldLanguage;
      }
    }

    // For backward compatibility.
    if (!IO.outputting()) {
      IO.mapOptional("AlignEscapedNewlinesLeft", Style.AlignEscapedNewlines);
      IO.mapOptional("DerivePointerBinding", Style.DerivePointerAlignment);
      IO.mapOptional("IndentFunctionDeclarationAfterType",
                     Style.IndentWrappedFunctionNames);
      IO.mapOptional("PointerBindsToType", Style.PointerAlignment);
      IO.mapOptional("SpaceAfterControlStatementKeyword",
                     Style.SpaceBeforeParens);
    }

    IO.mapOptional("AccessModifierOffset", Style.AccessModifierOffset);
    IO.mapOptional("AlignAfterOpenBracket", Style.AlignAfterOpenBracket);
    IO.mapOptional("AlignConsecutiveAssignments",
                   Style.AlignConsecutiveAssignments);
    IO.mapOptional("AlignConsecutiveDeclarations",
                   Style.AlignConsecutiveDeclarations);
    IO.mapOptional("AlignEscapedNewlines", Style.AlignEscapedNewlines);
    IO.mapOptional("AlignOperands", Style.AlignOperands);
    IO.mapOptional("AlignTrailingComments", Style.AlignTrailingComments);
    IO.mapOptional("AllowAllParametersOfDeclarationOnNextLine",
                   Style.AllowAllParametersOfDeclarationOnNextLine);
    IO.mapOptional("AllowShortBlocksOnASingleLine",
                   Style.AllowShortBlocksOnASingleLine);
    IO.mapOptional("AllowShortCaseLabelsOnASingleLine",
                   Style.AllowShortCaseLabelsOnASingleLine);
    IO.mapOptional("AllowShortFunctionsOnASingleLine",
                   Style.AllowShortFunctionsOnASingleLine);
    IO.mapOptional("AllowShortIfStatementsOnASingleLine",
                   Style.AllowShortIfStatementsOnASingleLine);
    IO.mapOptional("AllowShortLoopsOnASingleLine",
                   Style.AllowShortLoopsOnASingleLine);
    IO.mapOptional("AlwaysBreakAfterDefinitionReturnType",
                   Style.AlwaysBreakAfterDefinitionReturnType);
    IO.mapOptional("AlwaysBreakAfterReturnType",
                   Style.AlwaysBreakAfterReturnType);
    // If AlwaysBreakAfterDefinitionReturnType was specified but
    // AlwaysBreakAfterReturnType was not, initialize the latter from the
    // former for backwards compatibility.
    if (Style.AlwaysBreakAfterDefinitionReturnType != FormatStyle::DRTBS_None &&
        Style.AlwaysBreakAfterReturnType == FormatStyle::RTBS_None) {
      if (Style.AlwaysBreakAfterDefinitionReturnType == FormatStyle::DRTBS_All)
        Style.AlwaysBreakAfterReturnType = FormatStyle::RTBS_AllDefinitions;
      else if (Style.AlwaysBreakAfterDefinitionReturnType ==
               FormatStyle::DRTBS_TopLevel)
        Style.AlwaysBreakAfterReturnType =
            FormatStyle::RTBS_TopLevelDefinitions;
    }

    IO.mapOptional("AlwaysBreakBeforeMultilineStrings",
                   Style.AlwaysBreakBeforeMultilineStrings);
    IO.mapOptional("AlwaysBreakTemplateDeclarations",
                   Style.AlwaysBreakTemplateDeclarations);
    IO.mapOptional("BinPackArguments", Style.BinPackArguments);
    IO.mapOptional("BinPackParameters", Style.BinPackParameters);
    IO.mapOptional("BraceWrapping", Style.BraceWrapping);
    IO.mapOptional("BreakBeforeBinaryOperators",
                   Style.BreakBeforeBinaryOperators);
    IO.mapOptional("BreakBeforeBraces", Style.BreakBeforeBraces);
    IO.mapOptional("BreakBeforeInheritanceComma",
                   Style.BreakBeforeInheritanceComma);
    IO.mapOptional("BreakBeforeTernaryOperators",
                   Style.BreakBeforeTernaryOperators);

    bool BreakConstructorInitializersBeforeComma = false;
    IO.mapOptional("BreakConstructorInitializersBeforeComma",
                   BreakConstructorInitializersBeforeComma);
    IO.mapOptional("BreakConstructorInitializers",
                   Style.BreakConstructorInitializers);
    // If BreakConstructorInitializersBeforeComma was specified but
    // BreakConstructorInitializers was not, initialize the latter from the
    // former for backwards compatibility.
    if (BreakConstructorInitializersBeforeComma &&
        Style.BreakConstructorInitializers == FormatStyle::BCIS_BeforeColon)
      Style.BreakConstructorInitializers = FormatStyle::BCIS_BeforeComma;

    IO.mapOptional("BreakAfterJavaFieldAnnotations",
                   Style.BreakAfterJavaFieldAnnotations);
    IO.mapOptional("BreakStringLiterals", Style.BreakStringLiterals);
    IO.mapOptional("ColumnLimit", Style.ColumnLimit);
    IO.mapOptional("CommentPragmas", Style.CommentPragmas);
    IO.mapOptional("CompactNamespaces", Style.CompactNamespaces);
    IO.mapOptional("ConstructorInitializerAllOnOneLineOrOnePerLine",
                   Style.ConstructorInitializerAllOnOneLineOrOnePerLine);
    IO.mapOptional("ConstructorInitializerIndentWidth",
                   Style.ConstructorInitializerIndentWidth);
    IO.mapOptional("ContinuationIndentWidth", Style.ContinuationIndentWidth);
    IO.mapOptional("Cpp11BracedListStyle", Style.Cpp11BracedListStyle);
    IO.mapOptional("DerivePointerAlignment", Style.DerivePointerAlignment);
    IO.mapOptional("DisableFormat", Style.DisableFormat);
    IO.mapOptional("ExperimentalAutoDetectBinPacking",
                   Style.ExperimentalAutoDetectBinPacking);
    IO.mapOptional("FixNamespaceComments", Style.FixNamespaceComments);
    IO.mapOptional("ForEachMacros", Style.ForEachMacros);
    IO.mapOptional("IncludeBlocks", Style.IncludeBlocks);
    IO.mapOptional("IncludeCategories", Style.IncludeCategories);
    IO.mapOptional("IncludeIsMainRegex", Style.IncludeIsMainRegex);
    IO.mapOptional("IndentCaseLabels", Style.IndentCaseLabels);
    IO.mapOptional("IndentPPDirectives", Style.IndentPPDirectives);
    IO.mapOptional("IndentWidth", Style.IndentWidth);
    IO.mapOptional("IndentWrappedFunctionNames",
                   Style.IndentWrappedFunctionNames);
    IO.mapOptional("JavaScriptQuotes", Style.JavaScriptQuotes);
    IO.mapOptional("JavaScriptWrapImports", Style.JavaScriptWrapImports);
    IO.mapOptional("KeepEmptyLinesAtTheStartOfBlocks",
                   Style.KeepEmptyLinesAtTheStartOfBlocks);
    IO.mapOptional("MacroBlockBegin", Style.MacroBlockBegin);
    IO.mapOptional("MacroBlockEnd", Style.MacroBlockEnd);
    IO.mapOptional("MaxEmptyLinesToKeep", Style.MaxEmptyLinesToKeep);
    IO.mapOptional("NamespaceIndentation", Style.NamespaceIndentation);
    IO.mapOptional("ObjCBlockIndentWidth", Style.ObjCBlockIndentWidth);
    IO.mapOptional("ObjCSpaceAfterProperty", Style.ObjCSpaceAfterProperty);
    IO.mapOptional("ObjCSpaceBeforeProtocolList",
                   Style.ObjCSpaceBeforeProtocolList);
    IO.mapOptional("PenaltyBreakAssignment", Style.PenaltyBreakAssignment);
    IO.mapOptional("PenaltyBreakBeforeFirstCallParameter",
                   Style.PenaltyBreakBeforeFirstCallParameter);
    IO.mapOptional("PenaltyBreakComment", Style.PenaltyBreakComment);
    IO.mapOptional("PenaltyBreakFirstLessLess",
                   Style.PenaltyBreakFirstLessLess);
    IO.mapOptional("PenaltyBreakString", Style.PenaltyBreakString);
    IO.mapOptional("PenaltyExcessCharacter", Style.PenaltyExcessCharacter);
    IO.mapOptional("PenaltyReturnTypeOnItsOwnLine",
                   Style.PenaltyReturnTypeOnItsOwnLine);
    IO.mapOptional("PointerAlignment", Style.PointerAlignment);
    IO.mapOptional("RawStringFormats", Style.RawStringFormats);
    IO.mapOptional("ReflowComments", Style.ReflowComments);
    IO.mapOptional("SortIncludes", Style.SortIncludes);
    IO.mapOptional("SortUsingDeclarations", Style.SortUsingDeclarations);
    IO.mapOptional("SpaceAfterCStyleCast", Style.SpaceAfterCStyleCast);
    IO.mapOptional("SpaceAfterTemplateKeyword",
                   Style.SpaceAfterTemplateKeyword);
    IO.mapOptional("SpaceBeforeAssignmentOperators",
                   Style.SpaceBeforeAssignmentOperators);
    IO.mapOptional("SpaceBeforeParens", Style.SpaceBeforeParens);
    IO.mapOptional("SpaceInEmptyParentheses", Style.SpaceInEmptyParentheses);
    IO.mapOptional("SpacesBeforeTrailingComments",
                   Style.SpacesBeforeTrailingComments);
    IO.mapOptional("SpacesInAngles", Style.SpacesInAngles);
    IO.mapOptional("SpacesInContainerLiterals",
                   Style.SpacesInContainerLiterals);
    IO.mapOptional("SpacesInCStyleCastParentheses",
                   Style.SpacesInCStyleCastParentheses);
    IO.mapOptional("SpacesInParentheses", Style.SpacesInParentheses);
    IO.mapOptional("SpacesInSquareBrackets", Style.SpacesInSquareBrackets);
    IO.mapOptional("Standard", Style.Standard);
    IO.mapOptional("TabWidth", Style.TabWidth);
    IO.mapOptional("UseTab", Style.UseTab);
  }
};

template <> struct MappingTraits<FormatStyle::BraceWrappingFlags> {
  static void mapping(IO &IO, FormatStyle::BraceWrappingFlags &Wrapping) {
    IO.mapOptional("AfterClass", Wrapping.AfterClass);
    IO.mapOptional("AfterControlStatement", Wrapping.AfterControlStatement);
    IO.mapOptional("AfterEnum", Wrapping.AfterEnum);
    IO.mapOptional("AfterFunction", Wrapping.AfterFunction);
    IO.mapOptional("AfterNamespace", Wrapping.AfterNamespace);
    IO.mapOptional("AfterObjCDeclaration", Wrapping.AfterObjCDeclaration);
    IO.mapOptional("AfterStruct", Wrapping.AfterStruct);
    IO.mapOptional("AfterUnion", Wrapping.AfterUnion);
    IO.mapOptional("AfterExternBlock", Wrapping.AfterExternBlock);
    IO.mapOptional("BeforeCatch", Wrapping.BeforeCatch);
    IO.mapOptional("BeforeElse", Wrapping.BeforeElse);
    IO.mapOptional("IndentBraces", Wrapping.IndentBraces);
    IO.mapOptional("SplitEmptyFunction", Wrapping.SplitEmptyFunction);
    IO.mapOptional("SplitEmptyRecord", Wrapping.SplitEmptyRecord);
    IO.mapOptional("SplitEmptyNamespace", Wrapping.SplitEmptyNamespace);
  }
};

template <> struct MappingTraits<FormatStyle::IncludeCategory> {
  static void mapping(IO &IO, FormatStyle::IncludeCategory &Category) {
    IO.mapOptional("Regex", Category.Regex);
    IO.mapOptional("Priority", Category.Priority);
  }
};

template <> struct ScalarEnumerationTraits<FormatStyle::IncludeBlocksStyle> {
  static void enumeration(IO &IO, FormatStyle::IncludeBlocksStyle &Value) {
    IO.enumCase(Value, "Preserve", FormatStyle::IBS_Preserve);
    IO.enumCase(Value, "Merge", FormatStyle::IBS_Merge);
    IO.enumCase(Value, "Regroup", FormatStyle::IBS_Regroup);
  }
};

template <> struct MappingTraits<FormatStyle::RawStringFormat> {
  static void mapping(IO &IO, FormatStyle::RawStringFormat &Format) {
    IO.mapOptional("Delimiter", Format.Delimiter);
    IO.mapOptional("Language", Format.Language);
    IO.mapOptional("BasedOnStyle", Format.BasedOnStyle);
  }
};

// Allows to read vector<FormatStyle> while keeping default values.
// IO.getContext() should contain a pointer to the FormatStyle structure, that
// will be used to get default values for missing keys.
// If the first element has no Language specified, it will be treated as the
// default one for the following elements.
template <> struct DocumentListTraits<std::vector<FormatStyle>> {
  static size_t size(IO &IO, std::vector<FormatStyle> &Seq) {
    return Seq.size();
  }
  static FormatStyle &element(IO &IO, std::vector<FormatStyle> &Seq,
                              size_t Index) {
    if (Index >= Seq.size()) {
      assert(Index == Seq.size());
      FormatStyle Template;
      if (!Seq.empty() && Seq[0].Language == FormatStyle::LK_None) {
        Template = Seq[0];
      } else {
        Template = *((const FormatStyle *)IO.getContext());
        Template.Language = FormatStyle::LK_None;
      }
      Seq.resize(Index + 1, Template);
    }
    return Seq[Index];
  }
};
} // namespace yaml
} // namespace llvm

namespace clang {
namespace format {

const std::error_category &getParseCategory() {
  static ParseErrorCategory C;
  return C;
}
std::error_code make_error_code(ParseError e) {
  return std::error_code(static_cast<int>(e), getParseCategory());
}

inline llvm::Error make_string_error(const llvm::Twine &Message) {
  return llvm::make_error<llvm::StringError>(Message,
                                             llvm::inconvertibleErrorCode());
}

const char *ParseErrorCategory::name() const noexcept {
  return "clang-format.parse_error";
}

std::string ParseErrorCategory::message(int EV) const {
  switch (static_cast<ParseError>(EV)) {
  case ParseError::Success:
    return "Success";
  case ParseError::Error:
    return "Invalid argument";
  case ParseError::Unsuitable:
    return "Unsuitable";
  }
  llvm_unreachable("unexpected parse error");
}

static FormatStyle expandPresets(const FormatStyle &Style) {
  if (Style.BreakBeforeBraces == FormatStyle::BS_Custom)
    return Style;
  FormatStyle Expanded = Style;
  Expanded.BraceWrapping = {false, false, false, false, false,
                            false, false, false, false, false,
                            false, false, true,  true,  true};
  switch (Style.BreakBeforeBraces) {
  case FormatStyle::BS_Linux:
    Expanded.BraceWrapping.AfterClass = true;
    Expanded.BraceWrapping.AfterFunction = true;
    Expanded.BraceWrapping.AfterNamespace = true;
    break;
  case FormatStyle::BS_Mozilla:
    Expanded.BraceWrapping.AfterClass = true;
    Expanded.BraceWrapping.AfterEnum = true;
    Expanded.BraceWrapping.AfterFunction = true;
    Expanded.BraceWrapping.AfterStruct = true;
    Expanded.BraceWrapping.AfterUnion = true;
    Expanded.BraceWrapping.AfterExternBlock = true;
    Expanded.BraceWrapping.SplitEmptyFunction = true;
    Expanded.BraceWrapping.SplitEmptyRecord = false;
    break;
  case FormatStyle::BS_Stroustrup:
    Expanded.BraceWrapping.AfterFunction = true;
    Expanded.BraceWrapping.BeforeCatch = true;
    Expanded.BraceWrapping.BeforeElse = true;
    break;
  case FormatStyle::BS_Allman:
    Expanded.BraceWrapping.AfterClass = true;
    Expanded.BraceWrapping.AfterControlStatement = true;
    Expanded.BraceWrapping.AfterEnum = true;
    Expanded.BraceWrapping.AfterFunction = true;
    Expanded.BraceWrapping.AfterNamespace = true;
    Expanded.BraceWrapping.AfterObjCDeclaration = true;
    Expanded.BraceWrapping.AfterStruct = true;
    Expanded.BraceWrapping.AfterExternBlock = true;
    Expanded.BraceWrapping.BeforeCatch = true;
    Expanded.BraceWrapping.BeforeElse = true;
    break;
  case FormatStyle::BS_GNU:
    Expanded.BraceWrapping = {true, true, true, true, true, true, true, true,
                              true, true, true, true, true, true, true};
    break;
  case FormatStyle::BS_WebKit:
    Expanded.BraceWrapping.AfterFunction = true;
    break;
  default:
    break;
  }
  return Expanded;
}

FormatStyle getLLVMStyle() {
  FormatStyle LLVMStyle;
  LLVMStyle.Language = FormatStyle::LK_Cpp;
  LLVMStyle.AccessModifierOffset = -2;
  LLVMStyle.AlignEscapedNewlines = FormatStyle::ENAS_Right;
  LLVMStyle.AlignAfterOpenBracket = FormatStyle::BAS_Align;
  LLVMStyle.AlignOperands = true;
  LLVMStyle.AlignTrailingComments = true;
  LLVMStyle.AlignConsecutiveAssignments = false;
  LLVMStyle.AlignConsecutiveDeclarations = false;
  LLVMStyle.AllowAllParametersOfDeclarationOnNextLine = true;
  LLVMStyle.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_All;
  LLVMStyle.AllowShortBlocksOnASingleLine = false;
  LLVMStyle.AllowShortCaseLabelsOnASingleLine = false;
  LLVMStyle.AllowShortIfStatementsOnASingleLine = false;
  LLVMStyle.AllowShortLoopsOnASingleLine = false;
  LLVMStyle.AlwaysBreakAfterReturnType = FormatStyle::RTBS_None;
  LLVMStyle.AlwaysBreakAfterDefinitionReturnType = FormatStyle::DRTBS_None;
  LLVMStyle.AlwaysBreakBeforeMultilineStrings = false;
  LLVMStyle.AlwaysBreakTemplateDeclarations = false;
  LLVMStyle.BinPackArguments = true;
  LLVMStyle.BinPackParameters = true;
  LLVMStyle.BreakBeforeBinaryOperators = FormatStyle::BOS_None;
  LLVMStyle.BreakBeforeTernaryOperators = true;
  LLVMStyle.BreakBeforeBraces = FormatStyle::BS_Attach;
  LLVMStyle.BraceWrapping = {false, false, false, false, false,
                             false, false, false, false, false,
                             false, false, true,  true,  true};
  LLVMStyle.BreakAfterJavaFieldAnnotations = false;
  LLVMStyle.BreakConstructorInitializers = FormatStyle::BCIS_BeforeColon;
  LLVMStyle.BreakBeforeInheritanceComma = false;
  LLVMStyle.BreakStringLiterals = true;
  LLVMStyle.ColumnLimit = 80;
  LLVMStyle.CommentPragmas = "^ IWYU pragma:";
  LLVMStyle.CompactNamespaces = false;
  LLVMStyle.ConstructorInitializerAllOnOneLineOrOnePerLine = false;
  LLVMStyle.ConstructorInitializerIndentWidth = 4;
  LLVMStyle.ContinuationIndentWidth = 4;
  LLVMStyle.Cpp11BracedListStyle = true;
  LLVMStyle.DerivePointerAlignment = false;
  LLVMStyle.ExperimentalAutoDetectBinPacking = false;
  LLVMStyle.FixNamespaceComments = true;
  LLVMStyle.ForEachMacros.push_back("foreach");
  LLVMStyle.ForEachMacros.push_back("Q_FOREACH");
  LLVMStyle.ForEachMacros.push_back("BOOST_FOREACH");
  LLVMStyle.IncludeCategories = {{"^\"(llvm|llvm-c|clang|clang-c)/", 2},
                                 {"^(<|\"(gtest|gmock|isl|json)/)", 3},
                                 {".*", 1}};
  LLVMStyle.IncludeIsMainRegex = "(Test)?$";
  LLVMStyle.IncludeBlocks = FormatStyle::IBS_Preserve;
  LLVMStyle.IndentCaseLabels = false;
  LLVMStyle.IndentPPDirectives = FormatStyle::PPDIS_None;
  LLVMStyle.IndentWrappedFunctionNames = false;
  LLVMStyle.IndentWidth = 2;
  LLVMStyle.JavaScriptQuotes = FormatStyle::JSQS_Leave;
  LLVMStyle.JavaScriptWrapImports = true;
  LLVMStyle.TabWidth = 8;
  LLVMStyle.MaxEmptyLinesToKeep = 1;
  LLVMStyle.KeepEmptyLinesAtTheStartOfBlocks = true;
  LLVMStyle.NamespaceIndentation = FormatStyle::NI_None;
  LLVMStyle.ObjCBlockIndentWidth = 2;
  LLVMStyle.ObjCSpaceAfterProperty = false;
  LLVMStyle.ObjCSpaceBeforeProtocolList = true;
  LLVMStyle.PointerAlignment = FormatStyle::PAS_Right;
  LLVMStyle.SpacesBeforeTrailingComments = 1;
  LLVMStyle.Standard = FormatStyle::LS_Cpp11;
  LLVMStyle.UseTab = FormatStyle::UT_Never;
  LLVMStyle.RawStringFormats = {{"pb", FormatStyle::LK_TextProto, "google"}};
  LLVMStyle.ReflowComments = true;
  LLVMStyle.SpacesInParentheses = false;
  LLVMStyle.SpacesInSquareBrackets = false;
  LLVMStyle.SpaceInEmptyParentheses = false;
  LLVMStyle.SpacesInContainerLiterals = true;
  LLVMStyle.SpacesInCStyleCastParentheses = false;
  LLVMStyle.SpaceAfterCStyleCast = false;
  LLVMStyle.SpaceAfterTemplateKeyword = true;
  LLVMStyle.SpaceBeforeParens = FormatStyle::SBPO_ControlStatements;
  LLVMStyle.SpaceBeforeAssignmentOperators = true;
  LLVMStyle.SpacesInAngles = false;

  LLVMStyle.PenaltyBreakAssignment = prec::Assignment;
  LLVMStyle.PenaltyBreakComment = 300;
  LLVMStyle.PenaltyBreakFirstLessLess = 120;
  LLVMStyle.PenaltyBreakString = 1000;
  LLVMStyle.PenaltyExcessCharacter = 1000000;
  LLVMStyle.PenaltyReturnTypeOnItsOwnLine = 60;
  LLVMStyle.PenaltyBreakBeforeFirstCallParameter = 19;

  LLVMStyle.DisableFormat = false;
  LLVMStyle.SortIncludes = true;
  LLVMStyle.SortUsingDeclarations = true;

  return LLVMStyle;
}

FormatStyle getGoogleStyle(FormatStyle::LanguageKind Language) {
  if (Language == FormatStyle::LK_TextProto) {
    FormatStyle GoogleStyle = getGoogleStyle(FormatStyle::LK_Proto);
    GoogleStyle.Language = FormatStyle::LK_TextProto;
    return GoogleStyle;
  }

  FormatStyle GoogleStyle = getLLVMStyle();
  GoogleStyle.Language = Language;

  GoogleStyle.AccessModifierOffset = -1;
  GoogleStyle.AlignEscapedNewlines = FormatStyle::ENAS_Left;
  GoogleStyle.AllowShortIfStatementsOnASingleLine = true;
  GoogleStyle.AllowShortLoopsOnASingleLine = true;
  GoogleStyle.AlwaysBreakBeforeMultilineStrings = true;
  GoogleStyle.AlwaysBreakTemplateDeclarations = true;
  GoogleStyle.ConstructorInitializerAllOnOneLineOrOnePerLine = true;
  GoogleStyle.DerivePointerAlignment = true;
  GoogleStyle.IncludeCategories = {
      {"^<ext/.*\\.h>", 2}, {"^<.*\\.h>", 1}, {"^<.*", 2}, {".*", 3}};
  GoogleStyle.IncludeIsMainRegex = "([-_](test|unittest))?$";
  GoogleStyle.IndentCaseLabels = true;
  GoogleStyle.KeepEmptyLinesAtTheStartOfBlocks = false;
  GoogleStyle.ObjCSpaceAfterProperty = false;
  GoogleStyle.ObjCSpaceBeforeProtocolList = false;
  GoogleStyle.PointerAlignment = FormatStyle::PAS_Left;
  GoogleStyle.SpacesBeforeTrailingComments = 2;
  GoogleStyle.Standard = FormatStyle::LS_Auto;

  GoogleStyle.PenaltyReturnTypeOnItsOwnLine = 200;
  GoogleStyle.PenaltyBreakBeforeFirstCallParameter = 1;

  if (Language == FormatStyle::LK_Java) {
    GoogleStyle.AlignAfterOpenBracket = FormatStyle::BAS_DontAlign;
    GoogleStyle.AlignOperands = false;
    GoogleStyle.AlignTrailingComments = false;
    GoogleStyle.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_Empty;
    GoogleStyle.AllowShortIfStatementsOnASingleLine = false;
    GoogleStyle.AlwaysBreakBeforeMultilineStrings = false;
    GoogleStyle.BreakBeforeBinaryOperators = FormatStyle::BOS_NonAssignment;
    GoogleStyle.ColumnLimit = 100;
    GoogleStyle.SpaceAfterCStyleCast = true;
    GoogleStyle.SpacesBeforeTrailingComments = 1;
  } else if (Language == FormatStyle::LK_JavaScript) {
    GoogleStyle.AlignAfterOpenBracket = FormatStyle::BAS_AlwaysBreak;
    GoogleStyle.AlignOperands = false;
    GoogleStyle.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_Empty;
    GoogleStyle.AlwaysBreakBeforeMultilineStrings = false;
    GoogleStyle.BreakBeforeTernaryOperators = false;
    // taze:, triple slash directives (`/// <...`), @tag followed by { for a lot
    // of JSDoc tags, and @see, which is commonly followed by overlong URLs.
    GoogleStyle.CommentPragmas =
        "(taze:|^/[ \t]*<|(@[A-Za-z_0-9-]+[ \\t]*{)|@see)";
    GoogleStyle.MaxEmptyLinesToKeep = 3;
    GoogleStyle.NamespaceIndentation = FormatStyle::NI_All;
    GoogleStyle.SpacesInContainerLiterals = false;
    GoogleStyle.JavaScriptQuotes = FormatStyle::JSQS_Single;
    GoogleStyle.JavaScriptWrapImports = false;
  } else if (Language == FormatStyle::LK_Proto) {
    GoogleStyle.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_None;
    GoogleStyle.SpacesInContainerLiterals = false;
  } else if (Language == FormatStyle::LK_ObjC) {
    GoogleStyle.ColumnLimit = 100;
  }

  return GoogleStyle;
}

FormatStyle getChromiumStyle(FormatStyle::LanguageKind Language) {
  FormatStyle ChromiumStyle = getGoogleStyle(Language);
  if (Language == FormatStyle::LK_Java) {
    ChromiumStyle.AllowShortIfStatementsOnASingleLine = true;
    ChromiumStyle.BreakAfterJavaFieldAnnotations = true;
    ChromiumStyle.ContinuationIndentWidth = 8;
    ChromiumStyle.IndentWidth = 4;
  } else if (Language == FormatStyle::LK_JavaScript) {
    ChromiumStyle.AllowShortIfStatementsOnASingleLine = false;
    ChromiumStyle.AllowShortLoopsOnASingleLine = false;
  } else {
    ChromiumStyle.AllowAllParametersOfDeclarationOnNextLine = false;
    ChromiumStyle.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_Inline;
    ChromiumStyle.AllowShortIfStatementsOnASingleLine = false;
    ChromiumStyle.AllowShortLoopsOnASingleLine = false;
    ChromiumStyle.BinPackParameters = false;
    ChromiumStyle.DerivePointerAlignment = false;
    if (Language == FormatStyle::LK_ObjC)
      ChromiumStyle.ColumnLimit = 80;
  }
  return ChromiumStyle;
}

FormatStyle getMozillaStyle() {
  FormatStyle MozillaStyle = getLLVMStyle();
  MozillaStyle.AllowAllParametersOfDeclarationOnNextLine = false;
  MozillaStyle.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_Inline;
  MozillaStyle.AlwaysBreakAfterReturnType = FormatStyle::RTBS_TopLevel;
  MozillaStyle.AlwaysBreakAfterDefinitionReturnType =
      FormatStyle::DRTBS_TopLevel;
  MozillaStyle.AlwaysBreakTemplateDeclarations = true;
  MozillaStyle.BinPackParameters = false;
  MozillaStyle.BinPackArguments = false;
  MozillaStyle.BreakBeforeBraces = FormatStyle::BS_Mozilla;
  MozillaStyle.BreakConstructorInitializers = FormatStyle::BCIS_BeforeComma;
  MozillaStyle.BreakBeforeInheritanceComma = true;
  MozillaStyle.ConstructorInitializerIndentWidth = 2;
  MozillaStyle.ContinuationIndentWidth = 2;
  MozillaStyle.Cpp11BracedListStyle = false;
  MozillaStyle.FixNamespaceComments = false;
  MozillaStyle.IndentCaseLabels = true;
  MozillaStyle.ObjCSpaceAfterProperty = true;
  MozillaStyle.ObjCSpaceBeforeProtocolList = false;
  MozillaStyle.PenaltyReturnTypeOnItsOwnLine = 200;
  MozillaStyle.PointerAlignment = FormatStyle::PAS_Left;
  MozillaStyle.SpaceAfterTemplateKeyword = false;
  return MozillaStyle;
}

FormatStyle getWebKitStyle() {
  FormatStyle Style = getLLVMStyle();
  Style.AccessModifierOffset = -4;
  Style.AlignAfterOpenBracket = FormatStyle::BAS_DontAlign;
  Style.AlignOperands = false;
  Style.AlignTrailingComments = false;
  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_All;
  Style.BreakBeforeBraces = FormatStyle::BS_WebKit;
  Style.BreakConstructorInitializers = FormatStyle::BCIS_BeforeComma;
  Style.Cpp11BracedListStyle = false;
  Style.ColumnLimit = 0;
  Style.FixNamespaceComments = false;
  Style.IndentWidth = 4;
  Style.NamespaceIndentation = FormatStyle::NI_Inner;
  Style.ObjCBlockIndentWidth = 4;
  Style.ObjCSpaceAfterProperty = true;
  Style.PointerAlignment = FormatStyle::PAS_Left;
  return Style;
}

FormatStyle getGNUStyle() {
  FormatStyle Style = getLLVMStyle();
  Style.AlwaysBreakAfterDefinitionReturnType = FormatStyle::DRTBS_All;
  Style.AlwaysBreakAfterReturnType = FormatStyle::RTBS_AllDefinitions;
  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_All;
  Style.BreakBeforeBraces = FormatStyle::BS_GNU;
  Style.BreakBeforeTernaryOperators = true;
  Style.Cpp11BracedListStyle = false;
  Style.ColumnLimit = 79;
  Style.FixNamespaceComments = false;
  Style.SpaceBeforeParens = FormatStyle::SBPO_Always;
  Style.Standard = FormatStyle::LS_Cpp03;
  return Style;
}

FormatStyle getNoStyle() {
  FormatStyle NoStyle = getLLVMStyle();
  NoStyle.DisableFormat = true;
  NoStyle.SortIncludes = false;
  NoStyle.SortUsingDeclarations = false;
  return NoStyle;
}

bool getPredefinedStyle(StringRef Name, FormatStyle::LanguageKind Language,
                        FormatStyle *Style) {
  if (Name.equals_lower("llvm")) {
    *Style = getLLVMStyle();
  } else if (Name.equals_lower("chromium")) {
    *Style = getChromiumStyle(Language);
  } else if (Name.equals_lower("mozilla")) {
    *Style = getMozillaStyle();
  } else if (Name.equals_lower("google")) {
    *Style = getGoogleStyle(Language);
  } else if (Name.equals_lower("webkit")) {
    *Style = getWebKitStyle();
  } else if (Name.equals_lower("gnu")) {
    *Style = getGNUStyle();
  } else if (Name.equals_lower("none")) {
    *Style = getNoStyle();
  } else {
    return false;
  }

  Style->Language = Language;
  return true;
}

std::error_code parseConfiguration(StringRef Text, FormatStyle *Style) {
  assert(Style);
  FormatStyle::LanguageKind Language = Style->Language;
  assert(Language != FormatStyle::LK_None);
  if (Text.trim().empty())
    return make_error_code(ParseError::Error);

  std::vector<FormatStyle> Styles;
  llvm::yaml::Input Input(Text);
  // DocumentListTraits<vector<FormatStyle>> uses the context to get default
  // values for the fields, keys for which are missing from the configuration.
  // Mapping also uses the context to get the language to find the correct
  // base style.
  Input.setContext(Style);
  Input >> Styles;
  if (Input.error())
    return Input.error();

  for (unsigned i = 0; i < Styles.size(); ++i) {
    // Ensures that only the first configuration can skip the Language option.
    if (Styles[i].Language == FormatStyle::LK_None && i != 0)
      return make_error_code(ParseError::Error);
    // Ensure that each language is configured at most once.
    for (unsigned j = 0; j < i; ++j) {
      if (Styles[i].Language == Styles[j].Language) {
        DEBUG(llvm::dbgs()
              << "Duplicate languages in the config file on positions " << j
              << " and " << i << "\n");
        return make_error_code(ParseError::Error);
      }
    }
  }
  // Look for a suitable configuration starting from the end, so we can
  // find the configuration for the specific language first, and the default
  // configuration (which can only be at slot 0) after it.
  for (int i = Styles.size() - 1; i >= 0; --i) {
    if (Styles[i].Language == Language ||
        Styles[i].Language == FormatStyle::LK_None) {
      *Style = Styles[i];
      Style->Language = Language;
      return make_error_code(ParseError::Success);
    }
  }
  return make_error_code(ParseError::Unsuitable);
}

std::string configurationAsText(const FormatStyle &Style) {
  std::string Text;
  llvm::raw_string_ostream Stream(Text);
  llvm::yaml::Output Output(Stream);
  // We use the same mapping method for input and output, so we need a non-const
  // reference here.
  FormatStyle NonConstStyle = expandPresets(Style);
  Output << NonConstStyle;
  return Stream.str();
}

namespace {

class JavaScriptRequoter : public TokenAnalyzer {
public:
  JavaScriptRequoter(const Environment &Env, const FormatStyle &Style)
      : TokenAnalyzer(Env, Style) {}

  std::pair<tooling::Replacements, unsigned>
  analyze(TokenAnnotator &Annotator,
          SmallVectorImpl<AnnotatedLine *> &AnnotatedLines,
          FormatTokenLexer &Tokens) override {
    AffectedRangeMgr.computeAffectedLines(AnnotatedLines.begin(),
                                          AnnotatedLines.end());
    tooling::Replacements Result;
    requoteJSStringLiteral(AnnotatedLines, Result);
    return {Result, 0};
  }

private:
  // Replaces double/single-quoted string literal as appropriate, re-escaping
  // the contents in the process.
  void requoteJSStringLiteral(SmallVectorImpl<AnnotatedLine *> &Lines,
                              tooling::Replacements &Result) {
    for (AnnotatedLine *Line : Lines) {
      requoteJSStringLiteral(Line->Children, Result);
      if (!Line->Affected)
        continue;
      for (FormatToken *FormatTok = Line->First; FormatTok;
           FormatTok = FormatTok->Next) {
        StringRef Input = FormatTok->TokenText;
        if (FormatTok->Finalized || !FormatTok->isStringLiteral() ||
            // NB: testing for not starting with a double quote to avoid
            // breaking `template strings`.
            (Style.JavaScriptQuotes == FormatStyle::JSQS_Single &&
             !Input.startswith("\"")) ||
            (Style.JavaScriptQuotes == FormatStyle::JSQS_Double &&
             !Input.startswith("\'")))
          continue;

        // Change start and end quote.
        bool IsSingle = Style.JavaScriptQuotes == FormatStyle::JSQS_Single;
        SourceLocation Start = FormatTok->Tok.getLocation();
        auto Replace = [&](SourceLocation Start, unsigned Length,
                           StringRef ReplacementText) {
          auto Err = Result.add(tooling::Replacement(
              Env.getSourceManager(), Start, Length, ReplacementText));
          // FIXME: handle error. For now, print error message and skip the
          // replacement for release version.
          if (Err) {
            llvm::errs() << llvm::toString(std::move(Err)) << "\n";
            assert(false);
          }
        };
        Replace(Start, 1, IsSingle ? "'" : "\"");
        Replace(FormatTok->Tok.getEndLoc().getLocWithOffset(-1), 1,
                IsSingle ? "'" : "\"");

        // Escape internal quotes.
        bool Escaped = false;
        for (size_t i = 1; i < Input.size() - 1; i++) {
          switch (Input[i]) {
          case '\\':
            if (!Escaped && i + 1 < Input.size() &&
                ((IsSingle && Input[i + 1] == '"') ||
                 (!IsSingle && Input[i + 1] == '\''))) {
              // Remove this \, it's escaping a " or ' that no longer needs
              // escaping
              Replace(Start.getLocWithOffset(i), 1, "");
              continue;
            }
            Escaped = !Escaped;
            break;
          case '\"':
          case '\'':
            if (!Escaped && IsSingle == (Input[i] == '\'')) {
              // Escape the quote.
              Replace(Start.getLocWithOffset(i), 0, "\\");
            }
            Escaped = false;
            break;
          default:
            Escaped = false;
            break;
          }
        }
      }
    }
  }
};

class Formatter : public TokenAnalyzer {
public:
  Formatter(const Environment &Env, const FormatStyle &Style,
            FormattingAttemptStatus *Status)
      : TokenAnalyzer(Env, Style), Status(Status) {}

  std::pair<tooling::Replacements, unsigned>
  analyze(TokenAnnotator &Annotator,
          SmallVectorImpl<AnnotatedLine *> &AnnotatedLines,
          FormatTokenLexer &Tokens) override {
    tooling::Replacements Result;
    deriveLocalStyle(AnnotatedLines);
    AffectedRangeMgr.computeAffectedLines(AnnotatedLines.begin(),
                                          AnnotatedLines.end());
    for (unsigned i = 0, e = AnnotatedLines.size(); i != e; ++i) {
      Annotator.calculateFormattingInformation(*AnnotatedLines[i]);
    }
    Annotator.setCommentLineLevels(AnnotatedLines);

    WhitespaceManager Whitespaces(
        Env.getSourceManager(), Style,
        inputUsesCRLF(Env.getSourceManager().getBufferData(Env.getFileID())));
    ContinuationIndenter Indenter(Style, Tokens.getKeywords(),
                                  Env.getSourceManager(), Whitespaces, Encoding,
                                  BinPackInconclusiveFunctions);
    unsigned Penalty =
        UnwrappedLineFormatter(&Indenter, &Whitespaces, Style,
                               Tokens.getKeywords(), Env.getSourceManager(),
                               Status)
            .format(AnnotatedLines, /*DryRun=*/false,
                    /*AdditionalIndent=*/0,
                    /*FixBadIndentation=*/false,
                    /*FirstStartColumn=*/Env.getFirstStartColumn(),
                    /*NextStartColumn=*/Env.getNextStartColumn(),
                    /*LastStartColumn=*/Env.getLastStartColumn());
    for (const auto &R : Whitespaces.generateReplacements())
      if (Result.add(R))
        return std::make_pair(Result, 0);
    return std::make_pair(Result, Penalty);
  }

private:
  static bool inputUsesCRLF(StringRef Text) {
    return Text.count('\r') * 2 > Text.count('\n');
  }

  bool
  hasCpp03IncompatibleFormat(const SmallVectorImpl<AnnotatedLine *> &Lines) {
    for (const AnnotatedLine *Line : Lines) {
      if (hasCpp03IncompatibleFormat(Line->Children))
        return true;
      for (FormatToken *Tok = Line->First->Next; Tok; Tok = Tok->Next) {
        if (Tok->WhitespaceRange.getBegin() == Tok->WhitespaceRange.getEnd()) {
          if (Tok->is(tok::coloncolon) && Tok->Previous->is(TT_TemplateOpener))
            return true;
          if (Tok->is(TT_TemplateCloser) &&
              Tok->Previous->is(TT_TemplateCloser))
            return true;
        }
      }
    }
    return false;
  }

  int countVariableAlignments(const SmallVectorImpl<AnnotatedLine *> &Lines) {
    int AlignmentDiff = 0;
    for (const AnnotatedLine *Line : Lines) {
      AlignmentDiff += countVariableAlignments(Line->Children);
      for (FormatToken *Tok = Line->First; Tok && Tok->Next; Tok = Tok->Next) {
        if (!Tok->is(TT_PointerOrReference))
          continue;
        bool SpaceBefore =
            Tok->WhitespaceRange.getBegin() != Tok->WhitespaceRange.getEnd();
        bool SpaceAfter = Tok->Next->WhitespaceRange.getBegin() !=
                          Tok->Next->WhitespaceRange.getEnd();
        if (SpaceBefore && !SpaceAfter)
          ++AlignmentDiff;
        if (!SpaceBefore && SpaceAfter)
          --AlignmentDiff;
      }
    }
    return AlignmentDiff;
  }

  void
  deriveLocalStyle(const SmallVectorImpl<AnnotatedLine *> &AnnotatedLines) {
    bool HasBinPackedFunction = false;
    bool HasOnePerLineFunction = false;
    for (unsigned i = 0, e = AnnotatedLines.size(); i != e; ++i) {
      if (!AnnotatedLines[i]->First->Next)
        continue;
      FormatToken *Tok = AnnotatedLines[i]->First->Next;
      while (Tok->Next) {
        if (Tok->PackingKind == PPK_BinPacked)
          HasBinPackedFunction = true;
        if (Tok->PackingKind == PPK_OnePerLine)
          HasOnePerLineFunction = true;

        Tok = Tok->Next;
      }
    }
    if (Style.DerivePointerAlignment)
      Style.PointerAlignment = countVariableAlignments(AnnotatedLines) <= 0
                                   ? FormatStyle::PAS_Left
                                   : FormatStyle::PAS_Right;
    if (Style.Standard == FormatStyle::LS_Auto)
      Style.Standard = hasCpp03IncompatibleFormat(AnnotatedLines)
                           ? FormatStyle::LS_Cpp11
                           : FormatStyle::LS_Cpp03;
    BinPackInconclusiveFunctions =
        HasBinPackedFunction || !HasOnePerLineFunction;
  }

  bool BinPackInconclusiveFunctions;
  FormattingAttemptStatus *Status;
};

// This class clean up the erroneous/redundant code around the given ranges in
// file.
class Cleaner : public TokenAnalyzer {
public:
  Cleaner(const Environment &Env, const FormatStyle &Style)
      : TokenAnalyzer(Env, Style),
        DeletedTokens(FormatTokenLess(Env.getSourceManager())) {}

  // FIXME: eliminate unused parameters.
  std::pair<tooling::Replacements, unsigned>
  analyze(TokenAnnotator &Annotator,
          SmallVectorImpl<AnnotatedLine *> &AnnotatedLines,
          FormatTokenLexer &Tokens) override {
    // FIXME: in the current implementation the granularity of affected range
    // is an annotated line. However, this is not sufficient. Furthermore,
    // redundant code introduced by replacements does not necessarily
    // intercept with ranges of replacements that result in the redundancy.
    // To determine if some redundant code is actually introduced by
    // replacements(e.g. deletions), we need to come up with a more
    // sophisticated way of computing affected ranges.
    AffectedRangeMgr.computeAffectedLines(AnnotatedLines.begin(),
                                          AnnotatedLines.end());

    checkEmptyNamespace(AnnotatedLines);

    for (auto &Line : AnnotatedLines) {
      if (Line->Affected) {
        cleanupRight(Line->First, tok::comma, tok::comma);
        cleanupRight(Line->First, TT_CtorInitializerColon, tok::comma);
        cleanupRight(Line->First, tok::l_paren, tok::comma);
        cleanupLeft(Line->First, tok::comma, tok::r_paren);
        cleanupLeft(Line->First, TT_CtorInitializerComma, tok::l_brace);
        cleanupLeft(Line->First, TT_CtorInitializerColon, tok::l_brace);
        cleanupLeft(Line->First, TT_CtorInitializerColon, tok::equal);
      }
    }

    return {generateFixes(), 0};
  }

private:
  bool containsOnlyComments(const AnnotatedLine &Line) {
    for (FormatToken *Tok = Line.First; Tok != nullptr; Tok = Tok->Next) {
      if (Tok->isNot(tok::comment))
        return false;
    }
    return true;
  }

  // Iterate through all lines and remove any empty (nested) namespaces.
  void checkEmptyNamespace(SmallVectorImpl<AnnotatedLine *> &AnnotatedLines) {
    std::set<unsigned> DeletedLines;
    for (unsigned i = 0, e = AnnotatedLines.size(); i != e; ++i) {
      auto &Line = *AnnotatedLines[i];
      if (Line.startsWith(tok::kw_namespace) ||
          Line.startsWith(tok::kw_inline, tok::kw_namespace)) {
        checkEmptyNamespace(AnnotatedLines, i, i, DeletedLines);
      }
    }

    for (auto Line : DeletedLines) {
      FormatToken *Tok = AnnotatedLines[Line]->First;
      while (Tok) {
        deleteToken(Tok);
        Tok = Tok->Next;
      }
    }
  }

  // The function checks if the namespace, which starts from \p CurrentLine, and
  // its nested namespaces are empty and delete them if they are empty. It also
  // sets \p NewLine to the last line checked.
  // Returns true if the current namespace is empty.
  bool checkEmptyNamespace(SmallVectorImpl<AnnotatedLine *> &AnnotatedLines,
                           unsigned CurrentLine, unsigned &NewLine,
                           std::set<unsigned> &DeletedLines) {
    unsigned InitLine = CurrentLine, End = AnnotatedLines.size();
    if (Style.BraceWrapping.AfterNamespace) {
      // If the left brace is in a new line, we should consume it first so that
      // it does not make the namespace non-empty.
      // FIXME: error handling if there is no left brace.
      if (!AnnotatedLines[++CurrentLine]->startsWith(tok::l_brace)) {
        NewLine = CurrentLine;
        return false;
      }
    } else if (!AnnotatedLines[CurrentLine]->endsWith(tok::l_brace)) {
      return false;
    }
    while (++CurrentLine < End) {
      if (AnnotatedLines[CurrentLine]->startsWith(tok::r_brace))
        break;

      if (AnnotatedLines[CurrentLine]->startsWith(tok::kw_namespace) ||
          AnnotatedLines[CurrentLine]->startsWith(tok::kw_inline,
                                                  tok::kw_namespace)) {
        if (!checkEmptyNamespace(AnnotatedLines, CurrentLine, NewLine,
                                 DeletedLines))
          return false;
        CurrentLine = NewLine;
        continue;
      }

      if (containsOnlyComments(*AnnotatedLines[CurrentLine]))
        continue;

      // If there is anything other than comments or nested namespaces in the
      // current namespace, the namespace cannot be empty.
      NewLine = CurrentLine;
      return false;
    }

    NewLine = CurrentLine;
    if (CurrentLine >= End)
      return false;

    // Check if the empty namespace is actually affected by changed ranges.
    if (!AffectedRangeMgr.affectsCharSourceRange(CharSourceRange::getCharRange(
            AnnotatedLines[InitLine]->First->Tok.getLocation(),
            AnnotatedLines[CurrentLine]->Last->Tok.getEndLoc())))
      return false;

    for (unsigned i = InitLine; i <= CurrentLine; ++i) {
      DeletedLines.insert(i);
    }

    return true;
  }

  // Checks pairs {start, start->next},..., {end->previous, end} and deletes one
  // of the token in the pair if the left token has \p LK token kind and the
  // right token has \p RK token kind. If \p DeleteLeft is true, the left token
  // is deleted on match; otherwise, the right token is deleted.
  template <typename LeftKind, typename RightKind>
  void cleanupPair(FormatToken *Start, LeftKind LK, RightKind RK,
                   bool DeleteLeft) {
    auto NextNotDeleted = [this](const FormatToken &Tok) -> FormatToken * {
      for (auto *Res = Tok.Next; Res; Res = Res->Next)
        if (!Res->is(tok::comment) &&
            DeletedTokens.find(Res) == DeletedTokens.end())
          return Res;
      return nullptr;
    };
    for (auto *Left = Start; Left;) {
      auto *Right = NextNotDeleted(*Left);
      if (!Right)
        break;
      if (Left->is(LK) && Right->is(RK)) {
        deleteToken(DeleteLeft ? Left : Right);
        for (auto *Tok = Left->Next; Tok && Tok != Right; Tok = Tok->Next)
          deleteToken(Tok);
        // If the right token is deleted, we should keep the left token
        // unchanged and pair it with the new right token.
        if (!DeleteLeft)
          continue;
      }
      Left = Right;
    }
  }

  template <typename LeftKind, typename RightKind>
  void cleanupLeft(FormatToken *Start, LeftKind LK, RightKind RK) {
    cleanupPair(Start, LK, RK, /*DeleteLeft=*/true);
  }

  template <typename LeftKind, typename RightKind>
  void cleanupRight(FormatToken *Start, LeftKind LK, RightKind RK) {
    cleanupPair(Start, LK, RK, /*DeleteLeft=*/false);
  }

  // Delete the given token.
  inline void deleteToken(FormatToken *Tok) {
    if (Tok)
      DeletedTokens.insert(Tok);
  }

  tooling::Replacements generateFixes() {
    tooling::Replacements Fixes;
    std::vector<FormatToken *> Tokens;
    std::copy(DeletedTokens.begin(), DeletedTokens.end(),
              std::back_inserter(Tokens));

    // Merge multiple continuous token deletions into one big deletion so that
    // the number of replacements can be reduced. This makes computing affected
    // ranges more efficient when we run reformat on the changed code.
    unsigned Idx = 0;
    while (Idx < Tokens.size()) {
      unsigned St = Idx, End = Idx;
      while ((End + 1) < Tokens.size() &&
             Tokens[End]->Next == Tokens[End + 1]) {
        End++;
      }
      auto SR = CharSourceRange::getCharRange(Tokens[St]->Tok.getLocation(),
                                              Tokens[End]->Tok.getEndLoc());
      auto Err =
          Fixes.add(tooling::Replacement(Env.getSourceManager(), SR, ""));
      // FIXME: better error handling. for now just print error message and skip
      // for the release version.
      if (Err) {
        llvm::errs() << llvm::toString(std::move(Err)) << "\n";
        assert(false && "Fixes must not conflict!");
      }
      Idx = End + 1;
    }

    return Fixes;
  }

  // Class for less-than inequality comparason for the set `RedundantTokens`.
  // We store tokens in the order they appear in the translation unit so that
  // we do not need to sort them in `generateFixes()`.
  struct FormatTokenLess {
    FormatTokenLess(const SourceManager &SM) : SM(SM) {}

    bool operator()(const FormatToken *LHS, const FormatToken *RHS) const {
      return SM.isBeforeInTranslationUnit(LHS->Tok.getLocation(),
                                          RHS->Tok.getLocation());
    }
    const SourceManager &SM;
  };

  // Tokens to be deleted.
  std::set<FormatToken *, FormatTokenLess> DeletedTokens;
};

struct IncludeDirective {
  StringRef Filename;
  StringRef Text;
  unsigned Offset;
  int Category;
};

} // end anonymous namespace

// Determines whether 'Ranges' intersects with ('Start', 'End').
static bool affectsRange(ArrayRef<tooling::Range> Ranges, unsigned Start,
                         unsigned End) {
  for (auto Range : Ranges) {
    if (Range.getOffset() < End &&
        Range.getOffset() + Range.getLength() > Start)
      return true;
  }
  return false;
}

// Returns a pair (Index, OffsetToEOL) describing the position of the cursor
// before sorting/deduplicating. Index is the index of the include under the
// cursor in the original set of includes. If this include has duplicates, it is
// the index of the first of the duplicates as the others are going to be
// removed. OffsetToEOL describes the cursor's position relative to the end of
// its current line.
// If `Cursor` is not on any #include, `Index` will be UINT_MAX.
static std::pair<unsigned, unsigned>
FindCursorIndex(const SmallVectorImpl<IncludeDirective> &Includes,
                const SmallVectorImpl<unsigned> &Indices, unsigned Cursor) {
  unsigned CursorIndex = UINT_MAX;
  unsigned OffsetToEOL = 0;
  for (int i = 0, e = Includes.size(); i != e; ++i) {
    unsigned Start = Includes[Indices[i]].Offset;
    unsigned End = Start + Includes[Indices[i]].Text.size();
    if (!(Cursor >= Start && Cursor < End))
      continue;
    CursorIndex = Indices[i];
    OffsetToEOL = End - Cursor;
    // Put the cursor on the only remaining #include among the duplicate
    // #includes.
    while (--i >= 0 && Includes[CursorIndex].Text == Includes[Indices[i]].Text)
      CursorIndex = i;
    break;
  }
  return std::make_pair(CursorIndex, OffsetToEOL);
}

// Sorts and deduplicate a block of includes given by 'Includes' alphabetically
// adding the necessary replacement to 'Replaces'. 'Includes' must be in strict
// source order.
// #include directives with the same text will be deduplicated, and only the
// first #include in the duplicate #includes remains. If the `Cursor` is
// provided and put on a deleted #include, it will be moved to the remaining
// #include in the duplicate #includes.
static void sortCppIncludes(const FormatStyle &Style,
                            const SmallVectorImpl<IncludeDirective> &Includes,
                            ArrayRef<tooling::Range> Ranges, StringRef FileName,
                            tooling::Replacements &Replaces, unsigned *Cursor) {
  unsigned IncludesBeginOffset = Includes.front().Offset;
  unsigned IncludesEndOffset =
      Includes.back().Offset + Includes.back().Text.size();
  unsigned IncludesBlockSize = IncludesEndOffset - IncludesBeginOffset;
  if (!affectsRange(Ranges, IncludesBeginOffset, IncludesEndOffset))
    return;
  SmallVector<unsigned, 16> Indices;
  for (unsigned i = 0, e = Includes.size(); i != e; ++i)
    Indices.push_back(i);
  std::stable_sort(
      Indices.begin(), Indices.end(), [&](unsigned LHSI, unsigned RHSI) {
        return std::tie(Includes[LHSI].Category, Includes[LHSI].Filename) <
               std::tie(Includes[RHSI].Category, Includes[RHSI].Filename);
      });
  // The index of the include on which the cursor will be put after
  // sorting/deduplicating.
  unsigned CursorIndex;
  // The offset from cursor to the end of line.
  unsigned CursorToEOLOffset;
  if (Cursor)
    std::tie(CursorIndex, CursorToEOLOffset) =
        FindCursorIndex(Includes, Indices, *Cursor);

  // Deduplicate #includes.
  Indices.erase(std::unique(Indices.begin(), Indices.end(),
                            [&](unsigned LHSI, unsigned RHSI) {
                              return Includes[LHSI].Text == Includes[RHSI].Text;
                            }),
                Indices.end());

  int CurrentCategory = Includes.front().Category;

  // If the #includes are out of order, we generate a single replacement fixing
  // the entire block. Otherwise, no replacement is generated.
  if (Indices.size() == Includes.size() &&
      std::is_sorted(Indices.begin(), Indices.end()) &&
      Style.IncludeBlocks == FormatStyle::IBS_Preserve)
    return;

  std::string result;
  for (unsigned Index : Indices) {
    if (!result.empty()) {
      result += "\n";
      if (Style.IncludeBlocks == FormatStyle::IBS_Regroup &&
          CurrentCategory != Includes[Index].Category)
        result += "\n";
    }
    result += Includes[Index].Text;
    if (Cursor && CursorIndex == Index)
      *Cursor = IncludesBeginOffset + result.size() - CursorToEOLOffset;
    CurrentCategory = Includes[Index].Category;
  }

  auto Err = Replaces.add(tooling::Replacement(
      FileName, Includes.front().Offset, IncludesBlockSize, result));
  // FIXME: better error handling. For now, just skip the replacement for the
  // release version.
  if (Err) {
    llvm::errs() << llvm::toString(std::move(Err)) << "\n";
    assert(false);
  }
}

namespace {

// This class manages priorities of #include categories and calculates
// priorities for headers.
class IncludeCategoryManager {
public:
  IncludeCategoryManager(const FormatStyle &Style, StringRef FileName)
      : Style(Style), FileName(FileName) {
    FileStem = llvm::sys::path::stem(FileName);
    for (const auto &Category : Style.IncludeCategories)
      CategoryRegexs.emplace_back(Category.Regex, llvm::Regex::IgnoreCase);
    IsMainFile = FileName.endswith(".c") || FileName.endswith(".cc") ||
                 FileName.endswith(".cpp") || FileName.endswith(".c++") ||
                 FileName.endswith(".cxx") || FileName.endswith(".m") ||
                 FileName.endswith(".mm");
  }

  // Returns the priority of the category which \p IncludeName belongs to.
  // If \p CheckMainHeader is true and \p IncludeName is a main header, returns
  // 0. Otherwise, returns the priority of the matching category or INT_MAX.
  int getIncludePriority(StringRef IncludeName, bool CheckMainHeader) {
    int Ret = INT_MAX;
    for (unsigned i = 0, e = CategoryRegexs.size(); i != e; ++i)
      if (CategoryRegexs[i].match(IncludeName)) {
        Ret = Style.IncludeCategories[i].Priority;
        break;
      }
    if (CheckMainHeader && IsMainFile && Ret > 0 && isMainHeader(IncludeName))
      Ret = 0;
    return Ret;
  }

private:
  bool isMainHeader(StringRef IncludeName) const {
    if (!IncludeName.startswith("\""))
      return false;
    StringRef HeaderStem =
        llvm::sys::path::stem(IncludeName.drop_front(1).drop_back(1));
    if (FileStem.startswith(HeaderStem) ||
        FileStem.startswith_lower(HeaderStem)) {
      llvm::Regex MainIncludeRegex(
          (HeaderStem + Style.IncludeIsMainRegex).str(),
          llvm::Regex::IgnoreCase);
      if (MainIncludeRegex.match(FileStem))
        return true;
    }
    return false;
  }

  const FormatStyle &Style;
  bool IsMainFile;
  StringRef FileName;
  StringRef FileStem;
  SmallVector<llvm::Regex, 4> CategoryRegexs;
};

const char IncludeRegexPattern[] =
    R"(^[\t\ ]*#[\t\ ]*(import|include)[^"<]*(["<][^">]*[">]))";

} // anonymous namespace

tooling::Replacements sortCppIncludes(const FormatStyle &Style, StringRef Code,
                                      ArrayRef<tooling::Range> Ranges,
                                      StringRef FileName,
                                      tooling::Replacements &Replaces,
                                      unsigned *Cursor) {
  unsigned Prev = 0;
  unsigned SearchFrom = 0;
  llvm::Regex IncludeRegex(IncludeRegexPattern);
  SmallVector<StringRef, 4> Matches;
  SmallVector<IncludeDirective, 16> IncludesInBlock;

  // In compiled files, consider the first #include to be the main #include of
  // the file if it is not a system #include. This ensures that the header
  // doesn't have hidden dependencies
  // (http://llvm.org/docs/CodingStandards.html#include-style).
  //
  // FIXME: Do some sanity checking, e.g. edit distance of the base name, to fix
  // cases where the first #include is unlikely to be the main header.
  IncludeCategoryManager Categories(Style, FileName);
  bool FirstIncludeBlock = true;
  bool MainIncludeFound = false;
  bool FormattingOff = false;

  for (;;) {
    auto Pos = Code.find('\n', SearchFrom);
    StringRef Line =
        Code.substr(Prev, (Pos != StringRef::npos ? Pos : Code.size()) - Prev);

    StringRef Trimmed = Line.trim();
    if (Trimmed == "// clang-format off")
      FormattingOff = true;
    else if (Trimmed == "// clang-format on")
      FormattingOff = false;

    const bool EmptyLineSkipped =
        Trimmed.empty() && (Style.IncludeBlocks == FormatStyle::IBS_Merge ||
                            Style.IncludeBlocks == FormatStyle::IBS_Regroup);

    if (!FormattingOff && !Line.endswith("\\")) {
      if (IncludeRegex.match(Line, &Matches)) {
        StringRef IncludeName = Matches[2];
        int Category = Categories.getIncludePriority(
            IncludeName,
            /*CheckMainHeader=*/!MainIncludeFound && FirstIncludeBlock);
        if (Category == 0)
          MainIncludeFound = true;
        IncludesInBlock.push_back({IncludeName, Line, Prev, Category});
      } else if (!IncludesInBlock.empty() && !EmptyLineSkipped) {
        sortCppIncludes(Style, IncludesInBlock, Ranges, FileName, Replaces,
                        Cursor);
        IncludesInBlock.clear();
        FirstIncludeBlock = false;
      }
      Prev = Pos + 1;
    }
    if (Pos == StringRef::npos || Pos + 1 == Code.size())
      break;
    SearchFrom = Pos + 1;
  }
  if (!IncludesInBlock.empty())
    sortCppIncludes(Style, IncludesInBlock, Ranges, FileName, Replaces, Cursor);
  return Replaces;
}

bool isMpegTS(StringRef Code) {
  // MPEG transport streams use the ".ts" file extension. clang-format should
  // not attempt to format those. MPEG TS' frame format starts with 0x47 every
  // 189 bytes - detect that and return.
  return Code.size() > 188 && Code[0] == 0x47 && Code[188] == 0x47;
}

bool isLikelyXml(StringRef Code) { return Code.ltrim().startswith("<"); }

tooling::Replacements sortIncludes(const FormatStyle &Style, StringRef Code,
                                   ArrayRef<tooling::Range> Ranges,
                                   StringRef FileName, unsigned *Cursor) {
  tooling::Replacements Replaces;
  if (!Style.SortIncludes)
    return Replaces;
  if (isLikelyXml(Code))
    return Replaces;
  if (Style.Language == FormatStyle::LanguageKind::LK_JavaScript &&
      isMpegTS(Code))
    return Replaces;
  if (Style.Language == FormatStyle::LanguageKind::LK_JavaScript)
    return sortJavaScriptImports(Style, Code, Ranges, FileName);
  sortCppIncludes(Style, Code, Ranges, FileName, Replaces, Cursor);
  return Replaces;
}

template <typename T>
static llvm::Expected<tooling::Replacements>
processReplacements(T ProcessFunc, StringRef Code,
                    const tooling::Replacements &Replaces,
                    const FormatStyle &Style) {
  if (Replaces.empty())
    return tooling::Replacements();

  auto NewCode = applyAllReplacements(Code, Replaces);
  if (!NewCode)
    return NewCode.takeError();
  std::vector<tooling::Range> ChangedRanges = Replaces.getAffectedRanges();
  StringRef FileName = Replaces.begin()->getFilePath();

  tooling::Replacements FormatReplaces =
      ProcessFunc(Style, *NewCode, ChangedRanges, FileName);

  return Replaces.merge(FormatReplaces);
}

llvm::Expected<tooling::Replacements>
formatReplacements(StringRef Code, const tooling::Replacements &Replaces,
                   const FormatStyle &Style) {
  // We need to use lambda function here since there are two versions of
  // `sortIncludes`.
  auto SortIncludes = [](const FormatStyle &Style, StringRef Code,
                         std::vector<tooling::Range> Ranges,
                         StringRef FileName) -> tooling::Replacements {
    return sortIncludes(Style, Code, Ranges, FileName);
  };
  auto SortedReplaces =
      processReplacements(SortIncludes, Code, Replaces, Style);
  if (!SortedReplaces)
    return SortedReplaces.takeError();

  // We need to use lambda function here since there are two versions of
  // `reformat`.
  auto Reformat = [](const FormatStyle &Style, StringRef Code,
                     std::vector<tooling::Range> Ranges,
                     StringRef FileName) -> tooling::Replacements {
    return reformat(Style, Code, Ranges, FileName);
  };
  return processReplacements(Reformat, Code, *SortedReplaces, Style);
}

namespace {

inline bool isHeaderInsertion(const tooling::Replacement &Replace) {
  return Replace.getOffset() == UINT_MAX && Replace.getLength() == 0 &&
         llvm::Regex(IncludeRegexPattern).match(Replace.getReplacementText());
}

inline bool isHeaderDeletion(const tooling::Replacement &Replace) {
  return Replace.getOffset() == UINT_MAX && Replace.getLength() == 1;
}

// Returns the offset after skipping a sequence of tokens, matched by \p
// GetOffsetAfterSequence, from the start of the code.
// \p GetOffsetAfterSequence should be a function that matches a sequence of
// tokens and returns an offset after the sequence.
unsigned getOffsetAfterTokenSequence(
    StringRef FileName, StringRef Code, const FormatStyle &Style,
    llvm::function_ref<unsigned(const SourceManager &, Lexer &, Token &)>
        GetOffsetAfterSequence) {
  std::unique_ptr<Environment> Env =
      Environment::CreateVirtualEnvironment(Code, FileName, /*Ranges=*/{});
  const SourceManager &SourceMgr = Env->getSourceManager();
  Lexer Lex(Env->getFileID(), SourceMgr.getBuffer(Env->getFileID()), SourceMgr,
            getFormattingLangOpts(Style));
  Token Tok;
  // Get the first token.
  Lex.LexFromRawLexer(Tok);
  return GetOffsetAfterSequence(SourceMgr, Lex, Tok);
}

// Check if a sequence of tokens is like "#<Name> <raw_identifier>". If it is,
// \p Tok will be the token after this directive; otherwise, it can be any token
// after the given \p Tok (including \p Tok).
bool checkAndConsumeDirectiveWithName(Lexer &Lex, StringRef Name, Token &Tok) {
  bool Matched = Tok.is(tok::hash) && !Lex.LexFromRawLexer(Tok) &&
                 Tok.is(tok::raw_identifier) &&
                 Tok.getRawIdentifier() == Name && !Lex.LexFromRawLexer(Tok) &&
                 Tok.is(tok::raw_identifier);
  if (Matched)
    Lex.LexFromRawLexer(Tok);
  return Matched;
}

void skipComments(Lexer &Lex, Token &Tok) {
  while (Tok.is(tok::comment))
    if (Lex.LexFromRawLexer(Tok))
      return;
}

// Returns the offset after header guard directives and any comments
// before/after header guards. If no header guard presents in the code, this
// will returns the offset after skipping all comments from the start of the
// code.
unsigned getOffsetAfterHeaderGuardsAndComments(StringRef FileName,
                                               StringRef Code,
                                               const FormatStyle &Style) {
  return getOffsetAfterTokenSequence(
      FileName, Code, Style,
      [](const SourceManager &SM, Lexer &Lex, Token Tok) {
        skipComments(Lex, Tok);
        unsigned InitialOffset = SM.getFileOffset(Tok.getLocation());
        if (checkAndConsumeDirectiveWithName(Lex, "ifndef", Tok)) {
          skipComments(Lex, Tok);
          if (checkAndConsumeDirectiveWithName(Lex, "define", Tok))
            return SM.getFileOffset(Tok.getLocation());
        }
        return InitialOffset;
      });
}

// Check if a sequence of tokens is like
//    "#include ("header.h" | <header.h>)".
// If it is, \p Tok will be the token after this directive; otherwise, it can be
// any token after the given \p Tok (including \p Tok).
bool checkAndConsumeInclusiveDirective(Lexer &Lex, Token &Tok) {
  auto Matched = [&]() {
    Lex.LexFromRawLexer(Tok);
    return true;
  };
  if (Tok.is(tok::hash) && !Lex.LexFromRawLexer(Tok) &&
      Tok.is(tok::raw_identifier) && Tok.getRawIdentifier() == "include") {
    if (Lex.LexFromRawLexer(Tok))
      return false;
    if (Tok.is(tok::string_literal))
      return Matched();
    if (Tok.is(tok::less)) {
      while (!Lex.LexFromRawLexer(Tok) && Tok.isNot(tok::greater)) {
      }
      if (Tok.is(tok::greater))
        return Matched();
    }
  }
  return false;
}

// Returns the offset of the last #include directive after which a new
// #include can be inserted. This ignores #include's after the #include block(s)
// in the beginning of a file to avoid inserting headers into code sections
// where new #include's should not be added by default.
// These code sections include:
//      - raw string literals (containing #include).
//      - #if blocks.
//      - Special #include's among declarations (e.g. functions).
//
// If no #include after which a new #include can be inserted, this returns the
// offset after skipping all comments from the start of the code.
// Inserting after an #include is not allowed if it comes after code that is not
// #include (e.g. pre-processing directive that is not #include, declarations).
unsigned getMaxHeaderInsertionOffset(StringRef FileName, StringRef Code,
                                     const FormatStyle &Style) {
  return getOffsetAfterTokenSequence(
      FileName, Code, Style,
      [](const SourceManager &SM, Lexer &Lex, Token Tok) {
        skipComments(Lex, Tok);
        unsigned MaxOffset = SM.getFileOffset(Tok.getLocation());
        while (checkAndConsumeInclusiveDirective(Lex, Tok))
          MaxOffset = SM.getFileOffset(Tok.getLocation());
        return MaxOffset;
      });
}

bool isDeletedHeader(llvm::StringRef HeaderName,
                     const std::set<llvm::StringRef> &HeadersToDelete) {
  return HeadersToDelete.count(HeaderName) ||
         HeadersToDelete.count(HeaderName.trim("\"<>"));
}

// FIXME: insert empty lines between newly created blocks.
tooling::Replacements
fixCppIncludeInsertions(StringRef Code, const tooling::Replacements &Replaces,
                        const FormatStyle &Style) {
  if (!Style.isCpp())
    return Replaces;

  tooling::Replacements HeaderInsertions;
  std::set<llvm::StringRef> HeadersToDelete;
  tooling::Replacements Result;
  for (const auto &R : Replaces) {
    if (isHeaderInsertion(R)) {
      // Replacements from \p Replaces must be conflict-free already, so we can
      // simply consume the error.
      llvm::consumeError(HeaderInsertions.add(R));
    } else if (isHeaderDeletion(R)) {
      HeadersToDelete.insert(R.getReplacementText());
    } else if (R.getOffset() == UINT_MAX) {
      llvm::errs() << "Insertions other than header #include insertion are "
                      "not supported! "
                   << R.getReplacementText() << "\n";
    } else {
      llvm::consumeError(Result.add(R));
    }
  }
  if (HeaderInsertions.empty() && HeadersToDelete.empty())
    return Replaces;

  llvm::Regex IncludeRegex(IncludeRegexPattern);
  llvm::Regex DefineRegex(R"(^[\t\ ]*#[\t\ ]*define[\t\ ]*[^\\]*$)");
  SmallVector<StringRef, 4> Matches;

  StringRef FileName = Replaces.begin()->getFilePath();
  IncludeCategoryManager Categories(Style, FileName);

  // Record the offset of the end of the last include in each category.
  std::map<int, int> CategoryEndOffsets;
  // All possible priorities.
  // Add 0 for main header and INT_MAX for headers that are not in any category.
  std::set<int> Priorities = {0, INT_MAX};
  for (const auto &Category : Style.IncludeCategories)
    Priorities.insert(Category.Priority);
  int FirstIncludeOffset = -1;
  // All new headers should be inserted after this offset.
  unsigned MinInsertOffset =
      getOffsetAfterHeaderGuardsAndComments(FileName, Code, Style);
  StringRef TrimmedCode = Code.drop_front(MinInsertOffset);
  // Max insertion offset in the original code.
  unsigned MaxInsertOffset =
      MinInsertOffset +
      getMaxHeaderInsertionOffset(FileName, TrimmedCode, Style);
  SmallVector<StringRef, 32> Lines;
  TrimmedCode.split(Lines, '\n');
  unsigned Offset = MinInsertOffset;
  unsigned NextLineOffset;
  std::set<StringRef> ExistingIncludes;
  for (auto Line : Lines) {
    NextLineOffset = std::min(Code.size(), Offset + Line.size() + 1);
    if (IncludeRegex.match(Line, &Matches)) {
      // The header name with quotes or angle brackets.
      StringRef IncludeName = Matches[2];
      ExistingIncludes.insert(IncludeName);
      // Only record the offset of current #include if we can insert after it.
      if (Offset <= MaxInsertOffset) {
        int Category = Categories.getIncludePriority(
            IncludeName, /*CheckMainHeader=*/FirstIncludeOffset < 0);
        CategoryEndOffsets[Category] = NextLineOffset;
        if (FirstIncludeOffset < 0)
          FirstIncludeOffset = Offset;
      }
      if (isDeletedHeader(IncludeName, HeadersToDelete)) {
        // If this is the last line without trailing newline, we need to make
        // sure we don't delete across the file boundary.
        unsigned Length = std::min(Line.size() + 1, Code.size() - Offset);
        llvm::Error Err =
            Result.add(tooling::Replacement(FileName, Offset, Length, ""));
        if (Err) {
          // Ignore the deletion on conflict.
          llvm::errs() << "Failed to add header deletion replacement for "
                       << IncludeName << ": " << llvm::toString(std::move(Err))
                       << "\n";
        }
      }
    }
    Offset = NextLineOffset;
  }

  // Populate CategoryEndOfssets:
  // - Ensure that CategoryEndOffset[Highest] is always populated.
  // - If CategoryEndOffset[Priority] isn't set, use the next higher value that
  //   is set, up to CategoryEndOffset[Highest].
  auto Highest = Priorities.begin();
  if (CategoryEndOffsets.find(*Highest) == CategoryEndOffsets.end()) {
    if (FirstIncludeOffset >= 0)
      CategoryEndOffsets[*Highest] = FirstIncludeOffset;
    else
      CategoryEndOffsets[*Highest] = MinInsertOffset;
  }
  // By this point, CategoryEndOffset[Highest] is always set appropriately:
  //  - to an appropriate location before/after existing #includes, or
  //  - to right after the header guard, or
  //  - to the beginning of the file.
  for (auto I = ++Priorities.begin(), E = Priorities.end(); I != E; ++I)
    if (CategoryEndOffsets.find(*I) == CategoryEndOffsets.end())
      CategoryEndOffsets[*I] = CategoryEndOffsets[*std::prev(I)];

  bool NeedNewLineAtEnd = !Code.empty() && Code.back() != '\n';
  for (const auto &R : HeaderInsertions) {
    auto IncludeDirective = R.getReplacementText();
    bool Matched = IncludeRegex.match(IncludeDirective, &Matches);
    assert(Matched && "Header insertion replacement must have replacement text "
                      "'#include ...'");
    (void)Matched;
    auto IncludeName = Matches[2];
    if (ExistingIncludes.find(IncludeName) != ExistingIncludes.end()) {
      DEBUG(llvm::dbgs() << "Skip adding existing include : " << IncludeName
                         << "\n");
      continue;
    }
    int Category =
        Categories.getIncludePriority(IncludeName, /*CheckMainHeader=*/true);
    Offset = CategoryEndOffsets[Category];
    std::string NewInclude = !IncludeDirective.endswith("\n")
                                 ? (IncludeDirective + "\n").str()
                                 : IncludeDirective.str();
    // When inserting headers at end of the code, also append '\n' to the code
    // if it does not end with '\n'.
    if (NeedNewLineAtEnd && Offset == Code.size()) {
      NewInclude = "\n" + NewInclude;
      NeedNewLineAtEnd = false;
    }
    auto NewReplace = tooling::Replacement(FileName, Offset, 0, NewInclude);
    auto Err = Result.add(NewReplace);
    if (Err) {
      llvm::consumeError(std::move(Err));
      unsigned NewOffset = Result.getShiftedCodePosition(Offset);
      NewReplace = tooling::Replacement(FileName, NewOffset, 0, NewInclude);
      Result = Result.merge(tooling::Replacements(NewReplace));
    }
  }
  return Result;
}

} // anonymous namespace

llvm::Expected<tooling::Replacements>
cleanupAroundReplacements(StringRef Code, const tooling::Replacements &Replaces,
                          const FormatStyle &Style) {
  // We need to use lambda function here since there are two versions of
  // `cleanup`.
  auto Cleanup = [](const FormatStyle &Style, StringRef Code,
                    std::vector<tooling::Range> Ranges,
                    StringRef FileName) -> tooling::Replacements {
    return cleanup(Style, Code, Ranges, FileName);
  };
  // Make header insertion replacements insert new headers into correct blocks.
  tooling::Replacements NewReplaces =
      fixCppIncludeInsertions(Code, Replaces, Style);
  return processReplacements(Cleanup, Code, NewReplaces, Style);
}

namespace internal {
std::pair<tooling::Replacements, unsigned>
reformat(const FormatStyle &Style, StringRef Code,
         ArrayRef<tooling::Range> Ranges, unsigned FirstStartColumn,
         unsigned NextStartColumn, unsigned LastStartColumn, StringRef FileName,
         FormattingAttemptStatus *Status) {
  FormatStyle Expanded = expandPresets(Style);
  if (Expanded.DisableFormat)
    return {tooling::Replacements(), 0};
  if (isLikelyXml(Code))
    return {tooling::Replacements(), 0};
  if (Expanded.Language == FormatStyle::LK_JavaScript && isMpegTS(Code))
    return {tooling::Replacements(), 0};

  typedef std::function<std::pair<tooling::Replacements, unsigned>(
      const Environment &)>
      AnalyzerPass;
  SmallVector<AnalyzerPass, 4> Passes;

  if (Style.Language == FormatStyle::LK_Cpp) {
    if (Style.FixNamespaceComments)
      Passes.emplace_back([&](const Environment &Env) {
        return NamespaceEndCommentsFixer(Env, Expanded).process();
      });

    if (Style.SortUsingDeclarations)
      Passes.emplace_back([&](const Environment &Env) {
        return UsingDeclarationsSorter(Env, Expanded).process();
      });
  }

  if (Style.Language == FormatStyle::LK_JavaScript &&
      Style.JavaScriptQuotes != FormatStyle::JSQS_Leave)
    Passes.emplace_back([&](const Environment &Env) {
      return JavaScriptRequoter(Env, Expanded).process();
    });

  Passes.emplace_back([&](const Environment &Env) {
    return Formatter(Env, Expanded, Status).process();
  });

  std::unique_ptr<Environment> Env = Environment::CreateVirtualEnvironment(
      Code, FileName, Ranges, FirstStartColumn, NextStartColumn,
      LastStartColumn);
  llvm::Optional<std::string> CurrentCode = None;
  tooling::Replacements Fixes;
  unsigned Penalty = 0;
  for (size_t I = 0, E = Passes.size(); I < E; ++I) {
    std::pair<tooling::Replacements, unsigned> PassFixes = Passes[I](*Env);
    auto NewCode = applyAllReplacements(
        CurrentCode ? StringRef(*CurrentCode) : Code, PassFixes.first);
    if (NewCode) {
      Fixes = Fixes.merge(PassFixes.first);
      Penalty += PassFixes.second;
      if (I + 1 < E) {
        CurrentCode = std::move(*NewCode);
        Env = Environment::CreateVirtualEnvironment(
            *CurrentCode, FileName,
            tooling::calculateRangesAfterReplacements(Fixes, Ranges),
            FirstStartColumn, NextStartColumn, LastStartColumn);
      }
    }
  }

  return {Fixes, Penalty};
}
} // namespace internal

tooling::Replacements reformat(const FormatStyle &Style, StringRef Code,
                               ArrayRef<tooling::Range> Ranges,
                               StringRef FileName,
                               FormattingAttemptStatus *Status) {
  return internal::reformat(Style, Code, Ranges,
                            /*FirstStartColumn=*/0,
                            /*NextStartColumn=*/0,
                            /*LastStartColumn=*/0, FileName, Status)
      .first;
}

tooling::Replacements cleanup(const FormatStyle &Style, StringRef Code,
                              ArrayRef<tooling::Range> Ranges,
                              StringRef FileName) {
  // cleanups only apply to C++ (they mostly concern ctor commas etc.)
  if (Style.Language != FormatStyle::LK_Cpp)
    return tooling::Replacements();
  std::unique_ptr<Environment> Env =
      Environment::CreateVirtualEnvironment(Code, FileName, Ranges);
  Cleaner Clean(*Env, Style);
  return Clean.process().first;
}

tooling::Replacements reformat(const FormatStyle &Style, StringRef Code,
                               ArrayRef<tooling::Range> Ranges,
                               StringRef FileName, bool *IncompleteFormat) {
  FormattingAttemptStatus Status;
  auto Result = reformat(Style, Code, Ranges, FileName, &Status);
  if (!Status.FormatComplete)
    *IncompleteFormat = true;
  return Result;
}

tooling::Replacements fixNamespaceEndComments(const FormatStyle &Style,
                                              StringRef Code,
                                              ArrayRef<tooling::Range> Ranges,
                                              StringRef FileName) {
  std::unique_ptr<Environment> Env =
      Environment::CreateVirtualEnvironment(Code, FileName, Ranges);
  NamespaceEndCommentsFixer Fix(*Env, Style);
  return Fix.process().first;
}

tooling::Replacements sortUsingDeclarations(const FormatStyle &Style,
                                            StringRef Code,
                                            ArrayRef<tooling::Range> Ranges,
                                            StringRef FileName) {
  std::unique_ptr<Environment> Env =
      Environment::CreateVirtualEnvironment(Code, FileName, Ranges);
  UsingDeclarationsSorter Sorter(*Env, Style);
  return Sorter.process().first;
}

LangOptions getFormattingLangOpts(const FormatStyle &Style) {
  LangOptions LangOpts;
  LangOpts.CPlusPlus = 1;
  LangOpts.CPlusPlus11 = Style.Standard == FormatStyle::LS_Cpp03 ? 0 : 1;
  LangOpts.CPlusPlus14 = Style.Standard == FormatStyle::LS_Cpp03 ? 0 : 1;
  LangOpts.CPlusPlus17 = Style.Standard == FormatStyle::LS_Cpp03 ? 0 : 1;
  LangOpts.LineComment = 1;
  bool AlternativeOperators = Style.isCpp();
  LangOpts.CXXOperatorNames = AlternativeOperators ? 1 : 0;
  LangOpts.Bool = 1;
  LangOpts.ObjC1 = 1;
  LangOpts.ObjC2 = 1;
  LangOpts.MicrosoftExt = 1;    // To get kw___try, kw___finally.
  LangOpts.DeclSpecKeyword = 1; // To get __declspec.
  return LangOpts;
}

const char *StyleOptionHelpDescription =
    "Coding style, currently supports:\n"
    "  LLVM, Google, Chromium, Mozilla, WebKit.\n"
    "Use -style=file to load style configuration from\n"
    ".clang-format file located in one of the parent\n"
    "directories of the source file (or current\n"
    "directory for stdin).\n"
    "Use -style=\"{key: value, ...}\" to set specific\n"
    "parameters, e.g.:\n"
    "  -style=\"{BasedOnStyle: llvm, IndentWidth: 8}\"";

static FormatStyle::LanguageKind getLanguageByFileName(StringRef FileName) {
  if (FileName.endswith(".java"))
    return FormatStyle::LK_Java;
  if (FileName.endswith_lower(".js") || FileName.endswith_lower(".ts"))
    return FormatStyle::LK_JavaScript; // JavaScript or TypeScript.
  if (FileName.endswith(".m") || FileName.endswith(".mm"))
    return FormatStyle::LK_ObjC;
  if (FileName.endswith_lower(".proto") ||
      FileName.endswith_lower(".protodevel"))
    return FormatStyle::LK_Proto;
  if (FileName.endswith_lower(".textpb") ||
      FileName.endswith_lower(".pb.txt") ||
      FileName.endswith_lower(".textproto") ||
      FileName.endswith_lower(".asciipb"))
    return FormatStyle::LK_TextProto;
  if (FileName.endswith_lower(".td"))
    return FormatStyle::LK_TableGen;
  return FormatStyle::LK_Cpp;
}

llvm::Expected<FormatStyle> getStyle(StringRef StyleName, StringRef FileName,
                                     StringRef FallbackStyleName,
                                     StringRef Code, vfs::FileSystem *FS) {
  if (!FS) {
    FS = vfs::getRealFileSystem().get();
  }
  FormatStyle Style = getLLVMStyle();
  Style.Language = getLanguageByFileName(FileName);

  // This is a very crude detection of whether a header contains ObjC code that
  // should be improved over time and probably be done on tokens, not one the
  // bare content of the file.
  if (Style.Language == FormatStyle::LK_Cpp && FileName.endswith(".h") &&
      (Code.contains("\n- (") || Code.contains("\n+ (")))
    Style.Language = FormatStyle::LK_ObjC;

  FormatStyle FallbackStyle = getNoStyle();
  if (!getPredefinedStyle(FallbackStyleName, Style.Language, &FallbackStyle))
    return make_string_error("Invalid fallback style \"" + FallbackStyleName);

  if (StyleName.startswith("{")) {
    // Parse YAML/JSON style from the command line.
    if (std::error_code ec = parseConfiguration(StyleName, &Style))
      return make_string_error("Error parsing -style: " + ec.message());
    return Style;
  }

  if (!StyleName.equals_lower("file")) {
    if (!getPredefinedStyle(StyleName, Style.Language, &Style))
      return make_string_error("Invalid value for -style");
    return Style;
  }

  // Look for .clang-format/_clang-format file in the file's parent directories.
  SmallString<128> UnsuitableConfigFiles;
  SmallString<128> Path(FileName);
  if (std::error_code EC = FS->makeAbsolute(Path))
    return make_string_error(EC.message());

  for (StringRef Directory = Path; !Directory.empty();
       Directory = llvm::sys::path::parent_path(Directory)) {

    auto Status = FS->status(Directory);
    if (!Status ||
        Status->getType() != llvm::sys::fs::file_type::directory_file) {
      continue;
    }

    SmallString<128> ConfigFile(Directory);

    llvm::sys::path::append(ConfigFile, ".clang-format");
    DEBUG(llvm::dbgs() << "Trying " << ConfigFile << "...\n");

    Status = FS->status(ConfigFile.str());
    bool FoundConfigFile =
        Status && (Status->getType() == llvm::sys::fs::file_type::regular_file);
    if (!FoundConfigFile) {
      // Try _clang-format too, since dotfiles are not commonly used on Windows.
      ConfigFile = Directory;
      llvm::sys::path::append(ConfigFile, "_clang-format");
      DEBUG(llvm::dbgs() << "Trying " << ConfigFile << "...\n");
      Status = FS->status(ConfigFile.str());
      FoundConfigFile = Status && (Status->getType() ==
                                   llvm::sys::fs::file_type::regular_file);
    }

    if (FoundConfigFile) {
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Text =
          FS->getBufferForFile(ConfigFile.str());
      if (std::error_code EC = Text.getError())
        return make_string_error(EC.message());
      if (std::error_code ec =
              parseConfiguration(Text.get()->getBuffer(), &Style)) {
        if (ec == ParseError::Unsuitable) {
          if (!UnsuitableConfigFiles.empty())
            UnsuitableConfigFiles.append(", ");
          UnsuitableConfigFiles.append(ConfigFile);
          continue;
        }
        return make_string_error("Error reading " + ConfigFile + ": " +
                                 ec.message());
      }
      DEBUG(llvm::dbgs() << "Using configuration file " << ConfigFile << "\n");
      return Style;
    }
  }
  if (!UnsuitableConfigFiles.empty())
    return make_string_error("Configuration file(s) do(es) not support " +
                             getLanguageName(Style.Language) + ": " +
                             UnsuitableConfigFiles);
  return FallbackStyle;
}

} // namespace format
} // namespace clang
