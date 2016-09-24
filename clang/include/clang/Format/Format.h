//===--- Format.h - Format C++ code -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Various functions to configurably format source code.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FORMAT_FORMAT_H
#define LLVM_CLANG_FORMAT_FORMAT_H

#include "clang/Basic/LangOptions.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/ArrayRef.h"
#include <system_error>

namespace clang {

class Lexer;
class SourceManager;
class DiagnosticConsumer;

namespace vfs {
class FileSystem;
}

namespace format {

enum class ParseError { Success = 0, Error, Unsuitable };
class ParseErrorCategory final : public std::error_category {
public:
  const char *name() const LLVM_NOEXCEPT override;
  std::string message(int EV) const override;
};
const std::error_category &getParseCategory();
std::error_code make_error_code(ParseError e);

/// \brief The ``FormatStyle`` is used to configure the formatting to follow
/// specific guidelines.
struct FormatStyle {
  /// \brief The extra indent or outdent of access modifiers, e.g. ``public:``.
  int AccessModifierOffset;

  /// \brief Different styles for aligning after open brackets.
  enum BracketAlignmentStyle {
    /// \brief Align parameters on the open bracket, e.g.:
    /// \code
    ///   someLongFunction(argument1,
    ///                    argument2);
    /// \endcode
    BAS_Align,
    /// \brief Don't align, instead use ``ContinuationIndentWidth``, e.g.:
    /// \code
    ///   someLongFunction(argument1,
    ///       argument2);
    /// \endcode
    BAS_DontAlign,
    /// \brief Always break after an open bracket, if the parameters don't fit
    /// on a single line, e.g.:
    /// \code
    ///   someLongFunction(
    ///       argument1, argument2);
    /// \endcode
    BAS_AlwaysBreak,
  };

  /// \brief If ``true``, horizontally aligns arguments after an open bracket.
  ///
  /// This applies to round brackets (parentheses), angle brackets and square
  /// brackets.
  BracketAlignmentStyle AlignAfterOpenBracket;

  /// \brief If ``true``, aligns consecutive assignments.
  ///
  /// This will align the assignment operators of consecutive lines. This
  /// will result in formattings like
  /// \code
  ///   int aaaa = 12;
  ///   int b    = 23;
  ///   int ccc  = 23;
  /// \endcode
  bool AlignConsecutiveAssignments;

  /// \brief If ``true``, aligns consecutive declarations.
  ///
  /// This will align the declaration names of consecutive lines. This
  /// will result in formattings like
  /// \code
  ///   int         aaaa = 12;
  ///   float       b = 23;
  ///   std::string ccc = 23;
  /// \endcode
  bool AlignConsecutiveDeclarations;

  /// \brief If ``true``, aligns escaped newlines as far left as possible.
  /// Otherwise puts them into the right-most column.
  bool AlignEscapedNewlinesLeft;

  /// \brief If ``true``, horizontally align operands of binary and ternary
  /// expressions.
  ///
  /// Specifically, this aligns operands of a single expression that needs to be
  /// split over multiple lines, e.g.:
  /// \code
  ///   int aaa = bbbbbbbbbbbbbbb +
  ///             ccccccccccccccc;
  /// \endcode
  bool AlignOperands;

  /// \brief If ``true``, aligns trailing comments.
  bool AlignTrailingComments;

  /// \brief Allow putting all parameters of a function declaration onto
  /// the next line even if ``BinPackParameters`` is ``false``.
  bool AllowAllParametersOfDeclarationOnNextLine;

  /// \brief Allows contracting simple braced statements to a single line.
  ///
  /// E.g., this allows ``if (a) { return; }`` to be put on a single line.
  bool AllowShortBlocksOnASingleLine;

  /// \brief If ``true``, short case labels will be contracted to a single line.
  bool AllowShortCaseLabelsOnASingleLine;

  /// \brief Different styles for merging short functions containing at most one
  /// statement.
  enum ShortFunctionStyle {
    /// \brief Never merge functions into a single line.
    SFS_None,
    /// \brief Only merge empty functions.
    SFS_Empty,
    /// \brief Only merge functions defined inside a class. Implies "empty".
    SFS_Inline,
    /// \brief Merge all functions fitting on a single line.
    SFS_All,
  };

  /// \brief Dependent on the value, ``int f() { return 0; }`` can be put on a
  /// single line.
  ShortFunctionStyle AllowShortFunctionsOnASingleLine;

  /// \brief If ``true``, ``if (a) return;`` can be put on a single line.
  bool AllowShortIfStatementsOnASingleLine;

  /// \brief If ``true``, ``while (true) continue;`` can be put on a single
  /// line.
  bool AllowShortLoopsOnASingleLine;

  /// \brief Different ways to break after the function definition return type.
  enum DefinitionReturnTypeBreakingStyle {
    /// Break after return type automatically.
    /// ``PenaltyReturnTypeOnItsOwnLine`` is taken into account.
    DRTBS_None,
    /// Always break after the return type.
    DRTBS_All,
    /// Always break after the return types of top-level functions.
    DRTBS_TopLevel,
  };

  /// \brief Different ways to break after the function definition or
  /// declaration return type.
  enum ReturnTypeBreakingStyle {
    /// Break after return type automatically.
    /// ``PenaltyReturnTypeOnItsOwnLine`` is taken into account.
    RTBS_None,
    /// Always break after the return type.
    RTBS_All,
    /// Always break after the return types of top-level functions.
    RTBS_TopLevel,
    /// Always break after the return type of function definitions.
    RTBS_AllDefinitions,
    /// Always break after the return type of top-level definitions.
    RTBS_TopLevelDefinitions,
  };

  /// \brief The function definition return type breaking style to use.  This
  /// option is deprecated and is retained for backwards compatibility.
  DefinitionReturnTypeBreakingStyle AlwaysBreakAfterDefinitionReturnType;

  /// \brief The function declaration return type breaking style to use.
  ReturnTypeBreakingStyle AlwaysBreakAfterReturnType;

  /// \brief If ``true``, always break before multiline string literals.
  ///
  /// This flag is mean to make cases where there are multiple multiline strings
  /// in a file look more consistent. Thus, it will only take effect if wrapping
  /// the string at that point leads to it being indented
  /// ``ContinuationIndentWidth`` spaces from the start of the line.
  bool AlwaysBreakBeforeMultilineStrings;

  /// \brief If ``true``, always break after the ``template<...>`` of a template
  /// declaration.
  bool AlwaysBreakTemplateDeclarations;

  /// \brief If ``false``, a function call's arguments will either be all on the
  /// same line or will have one line each.
  bool BinPackArguments;

  /// \brief If ``false``, a function declaration's or function definition's
  /// parameters will either all be on the same line or will have one line each.
  bool BinPackParameters;

  /// \brief The style of breaking before or after binary operators.
  enum BinaryOperatorStyle {
    /// Break after operators.
    BOS_None,
    /// Break before operators that aren't assignments.
    BOS_NonAssignment,
    /// Break before operators.
    BOS_All,
  };

  /// \brief The way to wrap binary operators.
  BinaryOperatorStyle BreakBeforeBinaryOperators;

  /// \brief Different ways to attach braces to their surrounding context.
  enum BraceBreakingStyle {
    /// Always attach braces to surrounding context.
    BS_Attach,
    /// Like ``Attach``, but break before braces on function, namespace and
    /// class definitions.
    BS_Linux,
    /// Like ``Attach``, but break before braces on enum, function, and record
    /// definitions.
    BS_Mozilla,
    /// Like ``Attach``, but break before function definitions, ``catch``, and
    /// ``else``.
    BS_Stroustrup,
    /// Always break before braces.
    BS_Allman,
    /// Always break before braces and add an extra level of indentation to
    /// braces of control statements, not to those of class, function
    /// or other definitions.
    BS_GNU,
    /// Like ``Attach``, but break before functions.
    BS_WebKit,
    /// Configure each individual brace in `BraceWrapping`.
    BS_Custom
  };

  /// \brief The brace breaking style to use.
  BraceBreakingStyle BreakBeforeBraces;

  /// \brief Precise control over the wrapping of braces.
  struct BraceWrappingFlags {
    /// \brief Wrap class definitions.
    bool AfterClass;
    /// \brief Wrap control statements (``if``/``for``/``while``/``switch``/..).
    bool AfterControlStatement;
    /// \brief Wrap enum definitions.
    bool AfterEnum;
    /// \brief Wrap function definitions.
    bool AfterFunction;
    /// \brief Wrap namespace definitions.
    bool AfterNamespace;
    /// \brief Wrap ObjC definitions (``@autoreleasepool``, interfaces, ..).
    bool AfterObjCDeclaration;
    /// \brief Wrap struct definitions.
    bool AfterStruct;
    /// \brief Wrap union definitions.
    bool AfterUnion;
    /// \brief Wrap before ``catch``.
    bool BeforeCatch;
    /// \brief Wrap before ``else``.
    bool BeforeElse;
    /// \brief Indent the wrapped braces themselves.
    bool IndentBraces;
  };

  /// \brief Control of individual brace wrapping cases.
  ///
  /// If ``BreakBeforeBraces`` is set to ``BS_Custom``, use this to specify how
  /// each individual brace case should be handled. Otherwise, this is ignored.
  BraceWrappingFlags BraceWrapping;

  /// \brief If ``true``, ternary operators will be placed after line breaks.
  bool BreakBeforeTernaryOperators;

  /// \brief Always break constructor initializers before commas and align
  /// the commas with the colon.
  bool BreakConstructorInitializersBeforeComma;

  /// \brief Break after each annotation on a field in Java files.
  bool BreakAfterJavaFieldAnnotations;

  /// \brief Allow breaking string literals when formatting.
  bool BreakStringLiterals;

  /// \brief The column limit.
  ///
  /// A column limit of ``0`` means that there is no column limit. In this case,
  /// clang-format will respect the input's line breaking decisions within
  /// statements unless they contradict other rules.
  unsigned ColumnLimit;

  /// \brief A regular expression that describes comments with special meaning,
  /// which should not be split into lines or otherwise changed.
  std::string CommentPragmas;

  /// \brief If the constructor initializers don't fit on a line, put each
  /// initializer on its own line.
  bool ConstructorInitializerAllOnOneLineOrOnePerLine;

  /// \brief The number of characters to use for indentation of constructor
  /// initializer lists.
  unsigned ConstructorInitializerIndentWidth;

  /// \brief Indent width for line continuations.
  unsigned ContinuationIndentWidth;

  /// \brief If ``true``, format braced lists as best suited for C++11 braced
  /// lists.
  ///
  /// Important differences:
  /// - No spaces inside the braced list.
  /// - No line break before the closing brace.
  /// - Indentation with the continuation indent, not with the block indent.
  ///
  /// Fundamentally, C++11 braced lists are formatted exactly like function
  /// calls would be formatted in their place. If the braced list follows a name
  /// (e.g. a type or variable name), clang-format formats as if the ``{}`` were
  /// the parentheses of a function call with that name. If there is no name,
  /// a zero-length name is assumed.
  bool Cpp11BracedListStyle;

  /// \brief If ``true``, analyze the formatted file for the most common
  /// alignment of ``&`` and ``*``. ``PointerAlignment`` is then used only as
  /// fallback.
  bool DerivePointerAlignment;

  /// \brief Disables formatting completely.
  bool DisableFormat;

  /// \brief If ``true``, clang-format detects whether function calls and
  /// definitions are formatted with one parameter per line.
  ///
  /// Each call can be bin-packed, one-per-line or inconclusive. If it is
  /// inconclusive, e.g. completely on one line, but a decision needs to be
  /// made, clang-format analyzes whether there are other bin-packed cases in
  /// the input file and act accordingly.
  ///
  /// NOTE: This is an experimental flag, that might go away or be renamed. Do
  /// not use this in config files, etc. Use at your own risk.
  bool ExperimentalAutoDetectBinPacking;

  /// \brief A vector of macros that should be interpreted as foreach loops
  /// instead of as function calls.
  ///
  /// These are expected to be macros of the form:
  /// \code
  ///   FOREACH(<variable-declaration>, ...)
  ///     <loop-body>
  /// \endcode
  ///
  /// In the .clang-format configuration file, this can be configured like:
  /// \code{.yaml}
  ///   ForEachMacros: ['RANGES_FOR', 'FOREACH']
  /// \endcode
  ///
  /// For example: BOOST_FOREACH.
  std::vector<std::string> ForEachMacros;

  /// \brief See documentation of ``IncludeCategories``.
  struct IncludeCategory {
    /// \brief The regular expression that this category matches.
    std::string Regex;
    /// \brief The priority to assign to this category.
    int Priority;
    bool operator==(const IncludeCategory &Other) const {
      return Regex == Other.Regex && Priority == Other.Priority;
    }
  };

  /// \brief Regular expressions denoting the different ``#include`` categories
  /// used for ordering ``#includes``.
  ///
  /// These regular expressions are matched against the filename of an include
  /// (including the <> or "") in order. The value belonging to the first
  /// matching regular expression is assigned and ``#includes`` are sorted first
  /// according to increasing category number and then alphabetically within
  /// each category.
  ///
  /// If none of the regular expressions match, INT_MAX is assigned as
  /// category. The main header for a source file automatically gets category 0.
  /// so that it is generally kept at the beginning of the ``#includes``
  /// (http://llvm.org/docs/CodingStandards.html#include-style). However, you
  /// can also assign negative priorities if you have certain headers that
  /// always need to be first.
  ///
  /// To configure this in the .clang-format file, use:
  /// \code{.yaml}
  ///   IncludeCategories:
  ///     - Regex:           '^"(llvm|llvm-c|clang|clang-c)/'
  ///       Priority:        2
  ///     - Regex:           '^(<|"(gtest|isl|json)/)'
  ///       Priority:        3
  ///     - Regex:           '.*'
  ///       Priority:        1
  /// \endcode
  std::vector<IncludeCategory> IncludeCategories;

  /// \brief Specify a regular expression of suffixes that are allowed in the
  /// file-to-main-include mapping.
  ///
  /// When guessing whether a #include is the "main" include (to assign
  /// category 0, see above), use this regex of allowed suffixes to the header
  /// stem. A partial match is done, so that:
  /// - "" means "arbitrary suffix"
  /// - "$" means "no suffix"
  ///
  /// For example, if configured to "(_test)?$", then a header a.h would be seen
  /// as the "main" include in both a.cc and a_test.cc.
  std::string IncludeIsMainRegex;

  /// \brief Indent case labels one level from the switch statement.
  ///
  /// When ``false``, use the same indentation level as for the switch statement.
  /// Switch statement body is always indented one level more than case labels.
  bool IndentCaseLabels;

  /// \brief The number of columns to use for indentation.
  unsigned IndentWidth;

  /// \brief Indent if a function definition or declaration is wrapped after the
  /// type.
  bool IndentWrappedFunctionNames;

  /// \brief Quotation styles for JavaScript strings. Does not affect template
  /// strings.
  enum JavaScriptQuoteStyle {
    /// Leave string quotes as they are.
    JSQS_Leave,
    /// Always use single quotes.
    JSQS_Single,
    /// Always use double quotes.
    JSQS_Double
  };

  /// \brief The JavaScriptQuoteStyle to use for JavaScript strings.
  JavaScriptQuoteStyle JavaScriptQuotes;

  /// \brief Whether to wrap JavaScript import/export statements.
  bool JavaScriptWrapImports;

  /// \brief If true, empty lines at the start of blocks are kept.
  bool KeepEmptyLinesAtTheStartOfBlocks;

  /// \brief Supported languages.
  ///
  /// When stored in a configuration file, specifies the language, that the
  /// configuration targets. When passed to the ``reformat()`` function, enables
  /// syntax features specific to the language.
  enum LanguageKind {
    /// Do not use.
    LK_None,
    /// Should be used for C, C++, ObjectiveC, ObjectiveC++.
    LK_Cpp,
    /// Should be used for Java.
    LK_Java,
    /// Should be used for JavaScript.
    LK_JavaScript,
    /// Should be used for Protocol Buffers
    /// (https://developers.google.com/protocol-buffers/).
    LK_Proto,
    /// Should be used for TableGen code.
    LK_TableGen
  };

  /// \brief Language, this format style is targeted at.
  LanguageKind Language;

  /// \brief A regular expression matching macros that start a block.
  std::string MacroBlockBegin;

  /// \brief A regular expression matching macros that end a block.
  std::string MacroBlockEnd;

  /// \brief The maximum number of consecutive empty lines to keep.
  unsigned MaxEmptyLinesToKeep;

  /// \brief Different ways to indent namespace contents.
  enum NamespaceIndentationKind {
    /// Don't indent in namespaces.
    NI_None,
    /// Indent only in inner namespaces (nested in other namespaces).
    NI_Inner,
    /// Indent in all namespaces.
    NI_All
  };

  /// \brief The indentation used for namespaces.
  NamespaceIndentationKind NamespaceIndentation;

  /// \brief The number of characters to use for indentation of ObjC blocks.
  unsigned ObjCBlockIndentWidth;

  /// \brief Add a space after ``@property`` in Objective-C, i.e. use
  /// ``@property (readonly)`` instead of ``@property(readonly)``.
  bool ObjCSpaceAfterProperty;

  /// \brief Add a space in front of an Objective-C protocol list, i.e. use
  /// ``Foo <Protocol>`` instead of ``Foo<Protocol>``.
  bool ObjCSpaceBeforeProtocolList;

  /// \brief The penalty for breaking a function call after ``call(``.
  unsigned PenaltyBreakBeforeFirstCallParameter;

  /// \brief The penalty for each line break introduced inside a comment.
  unsigned PenaltyBreakComment;

  /// \brief The penalty for breaking before the first ``<<``.
  unsigned PenaltyBreakFirstLessLess;

  /// \brief The penalty for each line break introduced inside a string literal.
  unsigned PenaltyBreakString;

  /// \brief The penalty for each character outside of the column limit.
  unsigned PenaltyExcessCharacter;

  /// \brief Penalty for putting the return type of a function onto its own
  /// line.
  unsigned PenaltyReturnTypeOnItsOwnLine;

  /// \brief The ``&`` and ``*`` alignment style.
  enum PointerAlignmentStyle {
    /// Align pointer to the left.
    PAS_Left,
    /// Align pointer to the right.
    PAS_Right,
    /// Align pointer in the middle.
    PAS_Middle
  };

  /// \brief Pointer and reference alignment style.
  PointerAlignmentStyle PointerAlignment;

  /// \brief If ``true``, clang-format will attempt to re-flow comments.
  bool ReflowComments;

  /// \brief If ``true``, clang-format will sort ``#includes``.
  bool SortIncludes;

  /// \brief If ``true``, a space may be inserted after C style casts.
  bool SpaceAfterCStyleCast;

  /// \brief If \c true, a space will be inserted after the 'template' keyword.
  bool SpaceAfterTemplateKeyword;

  /// \brief If ``false``, spaces will be removed before assignment operators.
  bool SpaceBeforeAssignmentOperators;

  /// \brief Different ways to put a space before opening parentheses.
  enum SpaceBeforeParensOptions {
    /// Never put a space before opening parentheses.
    SBPO_Never,
    /// Put a space before opening parentheses only after control statement
    /// keywords (``for/if/while...``).
    SBPO_ControlStatements,
    /// Always put a space before opening parentheses, except when it's
    /// prohibited by the syntax rules (in function-like macro definitions) or
    /// when determined by other style rules (after unary operators, opening
    /// parentheses, etc.)
    SBPO_Always
  };

  /// \brief Defines in which cases to put a space before opening parentheses.
  SpaceBeforeParensOptions SpaceBeforeParens;

  /// \brief If ``true``, spaces may be inserted into ``()``.
  bool SpaceInEmptyParentheses;

  /// \brief The number of spaces before trailing line comments
  /// (``//`` - comments).
  ///
  /// This does not affect trailing block comments (``/*`` - comments) as
  /// those commonly have different usage patterns and a number of special
  /// cases.
  unsigned SpacesBeforeTrailingComments;

  /// \brief If ``true``, spaces will be inserted after ``<`` and before ``>``
  /// in template argument lists.
  bool SpacesInAngles;

  /// \brief If ``true``, spaces are inserted inside container literals (e.g.
  /// ObjC and Javascript array and dict literals).
  bool SpacesInContainerLiterals;

  /// \brief If ``true``, spaces may be inserted into C style casts.
  bool SpacesInCStyleCastParentheses;

  /// \brief If ``true``, spaces will be inserted after ``(`` and before ``)``.
  bool SpacesInParentheses;

  /// \brief If ``true``, spaces will be inserted after ``[`` and before ``]``.
  bool SpacesInSquareBrackets;

  /// \brief Supported language standards.
  enum LanguageStandard {
    /// Use C++03-compatible syntax.
    LS_Cpp03,
    /// Use features of C++11 (e.g. ``A<A<int>>`` instead of ``A<A<int> >``).
    LS_Cpp11,
    /// Automatic detection based on the input.
    LS_Auto
  };

  /// \brief Format compatible with this standard, e.g. use ``A<A<int> >``
  /// instead of ``A<A<int>>`` for ``LS_Cpp03``.
  LanguageStandard Standard;

  /// \brief The number of columns used for tab stops.
  unsigned TabWidth;

  /// \brief Different ways to use tab in formatting.
  enum UseTabStyle {
    /// Never use tab.
    UT_Never,
    /// Use tabs only for indentation.
    UT_ForIndentation,
    /// Use tabs only for line continuation and indentation.
    UT_ForContinuationAndIndentation,
    /// Use tabs whenever we need to fill whitespace that spans at least from
    /// one tab stop to the next one.
    UT_Always
  };

  /// \brief The way to use tab characters in the resulting file.
  UseTabStyle UseTab;

  bool operator==(const FormatStyle &R) const {
    return AccessModifierOffset == R.AccessModifierOffset &&
           AlignAfterOpenBracket == R.AlignAfterOpenBracket &&
           AlignConsecutiveAssignments == R.AlignConsecutiveAssignments &&
           AlignConsecutiveDeclarations == R.AlignConsecutiveDeclarations &&
           AlignEscapedNewlinesLeft == R.AlignEscapedNewlinesLeft &&
           AlignOperands == R.AlignOperands &&
           AlignTrailingComments == R.AlignTrailingComments &&
           AllowAllParametersOfDeclarationOnNextLine ==
               R.AllowAllParametersOfDeclarationOnNextLine &&
           AllowShortBlocksOnASingleLine == R.AllowShortBlocksOnASingleLine &&
           AllowShortCaseLabelsOnASingleLine ==
               R.AllowShortCaseLabelsOnASingleLine &&
           AllowShortFunctionsOnASingleLine ==
               R.AllowShortFunctionsOnASingleLine &&
           AllowShortIfStatementsOnASingleLine ==
               R.AllowShortIfStatementsOnASingleLine &&
           AllowShortLoopsOnASingleLine == R.AllowShortLoopsOnASingleLine &&
           AlwaysBreakAfterReturnType == R.AlwaysBreakAfterReturnType &&
           AlwaysBreakBeforeMultilineStrings ==
               R.AlwaysBreakBeforeMultilineStrings &&
           AlwaysBreakTemplateDeclarations ==
               R.AlwaysBreakTemplateDeclarations &&
           BinPackArguments == R.BinPackArguments &&
           BinPackParameters == R.BinPackParameters &&
           BreakBeforeBinaryOperators == R.BreakBeforeBinaryOperators &&
           BreakBeforeBraces == R.BreakBeforeBraces &&
           BreakBeforeTernaryOperators == R.BreakBeforeTernaryOperators &&
           BreakConstructorInitializersBeforeComma ==
               R.BreakConstructorInitializersBeforeComma &&
           BreakAfterJavaFieldAnnotations == R.BreakAfterJavaFieldAnnotations &&
           BreakStringLiterals == R.BreakStringLiterals &&
           ColumnLimit == R.ColumnLimit && CommentPragmas == R.CommentPragmas &&
           ConstructorInitializerAllOnOneLineOrOnePerLine ==
               R.ConstructorInitializerAllOnOneLineOrOnePerLine &&
           ConstructorInitializerIndentWidth ==
               R.ConstructorInitializerIndentWidth &&
           ContinuationIndentWidth == R.ContinuationIndentWidth &&
           Cpp11BracedListStyle == R.Cpp11BracedListStyle &&
           DerivePointerAlignment == R.DerivePointerAlignment &&
           DisableFormat == R.DisableFormat &&
           ExperimentalAutoDetectBinPacking ==
               R.ExperimentalAutoDetectBinPacking &&
           ForEachMacros == R.ForEachMacros &&
           IncludeCategories == R.IncludeCategories &&
           IndentCaseLabels == R.IndentCaseLabels &&
           IndentWidth == R.IndentWidth && Language == R.Language &&
           IndentWrappedFunctionNames == R.IndentWrappedFunctionNames &&
           JavaScriptQuotes == R.JavaScriptQuotes &&
           JavaScriptWrapImports == R.JavaScriptWrapImports &&
           KeepEmptyLinesAtTheStartOfBlocks ==
               R.KeepEmptyLinesAtTheStartOfBlocks &&
           MacroBlockBegin == R.MacroBlockBegin &&
           MacroBlockEnd == R.MacroBlockEnd &&
           MaxEmptyLinesToKeep == R.MaxEmptyLinesToKeep &&
           NamespaceIndentation == R.NamespaceIndentation &&
           ObjCBlockIndentWidth == R.ObjCBlockIndentWidth &&
           ObjCSpaceAfterProperty == R.ObjCSpaceAfterProperty &&
           ObjCSpaceBeforeProtocolList == R.ObjCSpaceBeforeProtocolList &&
           PenaltyBreakBeforeFirstCallParameter ==
               R.PenaltyBreakBeforeFirstCallParameter &&
           PenaltyBreakComment == R.PenaltyBreakComment &&
           PenaltyBreakFirstLessLess == R.PenaltyBreakFirstLessLess &&
           PenaltyBreakString == R.PenaltyBreakString &&
           PenaltyExcessCharacter == R.PenaltyExcessCharacter &&
           PenaltyReturnTypeOnItsOwnLine == R.PenaltyReturnTypeOnItsOwnLine &&
           PointerAlignment == R.PointerAlignment &&
           SpaceAfterCStyleCast == R.SpaceAfterCStyleCast &&
           SpaceAfterTemplateKeyword == R.SpaceAfterTemplateKeyword &&
           SpaceBeforeAssignmentOperators == R.SpaceBeforeAssignmentOperators &&
           SpaceBeforeParens == R.SpaceBeforeParens &&
           SpaceInEmptyParentheses == R.SpaceInEmptyParentheses &&
           SpacesBeforeTrailingComments == R.SpacesBeforeTrailingComments &&
           SpacesInAngles == R.SpacesInAngles &&
           SpacesInContainerLiterals == R.SpacesInContainerLiterals &&
           SpacesInCStyleCastParentheses == R.SpacesInCStyleCastParentheses &&
           SpacesInParentheses == R.SpacesInParentheses &&
           SpacesInSquareBrackets == R.SpacesInSquareBrackets &&
           Standard == R.Standard && TabWidth == R.TabWidth &&
           UseTab == R.UseTab;
  }
};

/// \brief Returns a format style complying with the LLVM coding standards:
/// http://llvm.org/docs/CodingStandards.html.
FormatStyle getLLVMStyle();

/// \brief Returns a format style complying with one of Google's style guides:
/// http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml.
/// http://google-styleguide.googlecode.com/svn/trunk/javascriptguide.xml.
/// https://developers.google.com/protocol-buffers/docs/style.
FormatStyle getGoogleStyle(FormatStyle::LanguageKind Language);

/// \brief Returns a format style complying with Chromium's style guide:
/// http://www.chromium.org/developers/coding-style.
FormatStyle getChromiumStyle(FormatStyle::LanguageKind Language);

/// \brief Returns a format style complying with Mozilla's style guide:
/// https://developer.mozilla.org/en-US/docs/Developer_Guide/Coding_Style.
FormatStyle getMozillaStyle();

/// \brief Returns a format style complying with Webkit's style guide:
/// http://www.webkit.org/coding/coding-style.html
FormatStyle getWebKitStyle();

/// \brief Returns a format style complying with GNU Coding Standards:
/// http://www.gnu.org/prep/standards/standards.html
FormatStyle getGNUStyle();

/// \brief Returns style indicating formatting should be not applied at all.
FormatStyle getNoStyle();

/// \brief Gets a predefined style for the specified language by name.
///
/// Currently supported names: LLVM, Google, Chromium, Mozilla. Names are
/// compared case-insensitively.
///
/// Returns ``true`` if the Style has been set.
bool getPredefinedStyle(StringRef Name, FormatStyle::LanguageKind Language,
                        FormatStyle *Style);

/// \brief Parse configuration from YAML-formatted text.
///
/// Style->Language is used to get the base style, if the ``BasedOnStyle``
/// option is present.
///
/// When ``BasedOnStyle`` is not present, options not present in the YAML
/// document, are retained in \p Style.
std::error_code parseConfiguration(StringRef Text, FormatStyle *Style);

/// \brief Gets configuration in a YAML string.
std::string configurationAsText(const FormatStyle &Style);

/// \brief Returns the replacements necessary to sort all ``#include`` blocks
/// that are affected by ``Ranges``.
tooling::Replacements sortIncludes(const FormatStyle &Style, StringRef Code,
                                   ArrayRef<tooling::Range> Ranges,
                                   StringRef FileName,
                                   unsigned *Cursor = nullptr);

/// \brief Returns the replacements corresponding to applying and formatting
/// \p Replaces on success; otheriwse, return an llvm::Error carrying
/// llvm::StringError.
llvm::Expected<tooling::Replacements>
formatReplacements(StringRef Code, const tooling::Replacements &Replaces,
                   const FormatStyle &Style);

/// \brief Returns the replacements corresponding to applying \p Replaces and
/// cleaning up the code after that on success; otherwise, return an llvm::Error
/// carrying llvm::StringError.
/// This also supports inserting/deleting C++ #include directives:
/// - If a replacement has offset UINT_MAX, length 0, and a replacement text
///   that is an #include directive, this will insert the #include into the
///   correct block in the \p Code.
/// - If a replacement has offset UINT_MAX, length 1, and a replacement text
///   that is the name of the header to be removed, the header will be removed
///   from \p Code if it exists.
llvm::Expected<tooling::Replacements>
cleanupAroundReplacements(StringRef Code, const tooling::Replacements &Replaces,
                          const FormatStyle &Style);

/// \brief Reformats the given \p Ranges in the file \p ID.
///
/// Each range is extended on either end to its next bigger logic unit, i.e.
/// everything that might influence its formatting or might be influenced by its
/// formatting.
///
/// Returns the ``Replacements`` necessary to make all \p Ranges comply with
/// \p Style.
///
/// If ``IncompleteFormat`` is non-null, its value will be set to true if any
/// of the affected ranges were not formatted due to a non-recoverable syntax
/// error.
tooling::Replacements reformat(const FormatStyle &Style,
                               SourceManager &SourceMgr, FileID ID,
                               ArrayRef<CharSourceRange> Ranges,
                               bool *IncompleteFormat = nullptr);

/// \brief Reformats the given \p Ranges in \p Code.
///
/// Otherwise identical to the reformat() function using a file ID.
tooling::Replacements reformat(const FormatStyle &Style, StringRef Code,
                               ArrayRef<tooling::Range> Ranges,
                               StringRef FileName = "<stdin>",
                               bool *IncompleteFormat = nullptr);

/// \brief Clean up any erroneous/redundant code in the given \p Ranges in the
/// file \p ID.
///
/// Returns the ``Replacements`` that clean up all \p Ranges in the file \p ID.
tooling::Replacements cleanup(const FormatStyle &Style,
                              SourceManager &SourceMgr, FileID ID,
                              ArrayRef<CharSourceRange> Ranges);

/// \brief Clean up any erroneous/redundant code in the given \p Ranges in \p
/// Code.
///
/// Otherwise identical to the cleanup() function using a file ID.
tooling::Replacements cleanup(const FormatStyle &Style, StringRef Code,
                              ArrayRef<tooling::Range> Ranges,
                              StringRef FileName = "<stdin>");

/// \brief Returns the ``LangOpts`` that the formatter expects you to set.
///
/// \param Style determines specific settings for lexing mode.
LangOptions getFormattingLangOpts(const FormatStyle &Style = getLLVMStyle());

/// \brief Description to be used for help text for a ``llvm::cl`` option for
/// specifying format style. The description is closely related to the operation
/// of ``getStyle()``.
extern const char *StyleOptionHelpDescription;

/// \brief Construct a FormatStyle based on ``StyleName``.
///
/// ``StyleName`` can take several forms:
/// * "{<key>: <value>, ...}" - Set specic style parameters.
/// * "<style name>" - One of the style names supported by
/// getPredefinedStyle().
/// * "file" - Load style configuration from a file called ``.clang-format``
/// located in one of the parent directories of ``FileName`` or the current
/// directory if ``FileName`` is empty.
///
/// \param[in] StyleName Style name to interpret according to the description
/// above.
/// \param[in] FileName Path to start search for .clang-format if ``StyleName``
/// == "file".
/// \param[in] FallbackStyle The name of a predefined style used to fallback to
/// in case the style can't be determined from \p StyleName.
/// \param[in] FS The underlying file system, in which the file resides. By
/// default, the file system is the real file system.
///
/// \returns FormatStyle as specified by ``StyleName``. If no style could be
/// determined, the default is LLVM Style (see ``getLLVMStyle()``).
FormatStyle getStyle(StringRef StyleName, StringRef FileName,
                     StringRef FallbackStyle, vfs::FileSystem *FS = nullptr);

// \brief Returns a string representation of ``Language``.
inline StringRef getLanguageName(FormatStyle::LanguageKind Language) {
  switch (Language) {
  case FormatStyle::LK_Cpp:
    return "C++";
  case FormatStyle::LK_Java:
    return "Java";
  case FormatStyle::LK_JavaScript:
    return "JavaScript";
  case FormatStyle::LK_Proto:
    return "Proto";
  default:
    return "Unknown";
  }
}

} // end namespace format
} // end namespace clang

namespace std {
template <>
struct is_error_code_enum<clang::format::ParseError> : std::true_type {};
}

#endif // LLVM_CLANG_FORMAT_FORMAT_H
