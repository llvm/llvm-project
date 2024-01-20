//===- unittest/Format/ConfigParseTest.cpp - Config parsing unit tests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Format/Format.h"

#include "llvm/Support/VirtualFileSystem.h"
#include "gtest/gtest.h"

namespace clang {
namespace format {
namespace {

FormatStyle getGoogleStyle() { return getGoogleStyle(FormatStyle::LK_Cpp); }

#define EXPECT_ALL_STYLES_EQUAL(Styles)                                        \
  for (size_t i = 1; i < Styles.size(); ++i)                                   \
  EXPECT_EQ(Styles[0], Styles[i])                                              \
      << "Style #" << i << " of " << Styles.size() << " differs from Style #0"

TEST(ConfigParseTest, GetsPredefinedStyleByName) {
  SmallVector<FormatStyle, 3> Styles;
  Styles.resize(3);

  Styles[0] = getLLVMStyle();
  EXPECT_TRUE(getPredefinedStyle("LLVM", FormatStyle::LK_Cpp, &Styles[1]));
  EXPECT_TRUE(getPredefinedStyle("lLvM", FormatStyle::LK_Cpp, &Styles[2]));
  EXPECT_ALL_STYLES_EQUAL(Styles);

  Styles[0] = getGoogleStyle();
  EXPECT_TRUE(getPredefinedStyle("Google", FormatStyle::LK_Cpp, &Styles[1]));
  EXPECT_TRUE(getPredefinedStyle("gOOgle", FormatStyle::LK_Cpp, &Styles[2]));
  EXPECT_ALL_STYLES_EQUAL(Styles);

  Styles[0] = getGoogleStyle(FormatStyle::LK_JavaScript);
  EXPECT_TRUE(
      getPredefinedStyle("Google", FormatStyle::LK_JavaScript, &Styles[1]));
  EXPECT_TRUE(
      getPredefinedStyle("gOOgle", FormatStyle::LK_JavaScript, &Styles[2]));
  EXPECT_ALL_STYLES_EQUAL(Styles);

  Styles[0] = getChromiumStyle(FormatStyle::LK_Cpp);
  EXPECT_TRUE(getPredefinedStyle("Chromium", FormatStyle::LK_Cpp, &Styles[1]));
  EXPECT_TRUE(getPredefinedStyle("cHRoMiUM", FormatStyle::LK_Cpp, &Styles[2]));
  EXPECT_ALL_STYLES_EQUAL(Styles);

  Styles[0] = getMozillaStyle();
  EXPECT_TRUE(getPredefinedStyle("Mozilla", FormatStyle::LK_Cpp, &Styles[1]));
  EXPECT_TRUE(getPredefinedStyle("moZILla", FormatStyle::LK_Cpp, &Styles[2]));
  EXPECT_ALL_STYLES_EQUAL(Styles);

  Styles[0] = getWebKitStyle();
  EXPECT_TRUE(getPredefinedStyle("WebKit", FormatStyle::LK_Cpp, &Styles[1]));
  EXPECT_TRUE(getPredefinedStyle("wEbKit", FormatStyle::LK_Cpp, &Styles[2]));
  EXPECT_ALL_STYLES_EQUAL(Styles);

  Styles[0] = getGNUStyle();
  EXPECT_TRUE(getPredefinedStyle("GNU", FormatStyle::LK_Cpp, &Styles[1]));
  EXPECT_TRUE(getPredefinedStyle("gnU", FormatStyle::LK_Cpp, &Styles[2]));
  EXPECT_ALL_STYLES_EQUAL(Styles);

  Styles[0] = getClangFormatStyle();
  EXPECT_TRUE(
      getPredefinedStyle("clang-format", FormatStyle::LK_Cpp, &Styles[1]));
  EXPECT_TRUE(
      getPredefinedStyle("Clang-format", FormatStyle::LK_Cpp, &Styles[2]));
  EXPECT_ALL_STYLES_EQUAL(Styles);

  EXPECT_FALSE(getPredefinedStyle("qwerty", FormatStyle::LK_Cpp, &Styles[0]));
}

TEST(ConfigParseTest, GetsCorrectBasedOnStyle) {
  SmallVector<FormatStyle, 8> Styles;
  Styles.resize(2);

  Styles[0] = getGoogleStyle();
  Styles[1] = getLLVMStyle();
  EXPECT_EQ(0, parseConfiguration("BasedOnStyle: Google", &Styles[1]).value());
  EXPECT_ALL_STYLES_EQUAL(Styles);

  Styles.resize(5);
  Styles[0] = getGoogleStyle(FormatStyle::LK_JavaScript);
  Styles[1] = getLLVMStyle();
  Styles[1].Language = FormatStyle::LK_JavaScript;
  EXPECT_EQ(0, parseConfiguration("BasedOnStyle: Google", &Styles[1]).value());

  Styles[2] = getLLVMStyle();
  Styles[2].Language = FormatStyle::LK_JavaScript;
  EXPECT_EQ(0, parseConfiguration("Language: JavaScript\n"
                                  "BasedOnStyle: Google",
                                  &Styles[2])
                   .value());

  Styles[3] = getLLVMStyle();
  Styles[3].Language = FormatStyle::LK_JavaScript;
  EXPECT_EQ(0, parseConfiguration("BasedOnStyle: Google\n"
                                  "Language: JavaScript",
                                  &Styles[3])
                   .value());

  Styles[4] = getLLVMStyle();
  Styles[4].Language = FormatStyle::LK_JavaScript;
  EXPECT_EQ(0, parseConfiguration("---\n"
                                  "BasedOnStyle: LLVM\n"
                                  "IndentWidth: 123\n"
                                  "---\n"
                                  "BasedOnStyle: Google\n"
                                  "Language: JavaScript",
                                  &Styles[4])
                   .value());
  EXPECT_ALL_STYLES_EQUAL(Styles);
}

#define CHECK_PARSE_BOOL_FIELD(FIELD, CONFIG_NAME)                             \
  Style.FIELD = false;                                                         \
  EXPECT_EQ(0, parseConfiguration(CONFIG_NAME ": true", &Style).value());      \
  EXPECT_TRUE(Style.FIELD);                                                    \
  EXPECT_EQ(0, parseConfiguration(CONFIG_NAME ": false", &Style).value());     \
  EXPECT_FALSE(Style.FIELD)

#define CHECK_PARSE_BOOL(FIELD) CHECK_PARSE_BOOL_FIELD(FIELD, #FIELD)

#define CHECK_PARSE_NESTED_BOOL_FIELD(STRUCT, FIELD, CONFIG_NAME)              \
  Style.STRUCT.FIELD = false;                                                  \
  EXPECT_EQ(0,                                                                 \
            parseConfiguration(#STRUCT ":\n  " CONFIG_NAME ": true", &Style)   \
                .value());                                                     \
  EXPECT_TRUE(Style.STRUCT.FIELD);                                             \
  EXPECT_EQ(0,                                                                 \
            parseConfiguration(#STRUCT ":\n  " CONFIG_NAME ": false", &Style)  \
                .value());                                                     \
  EXPECT_FALSE(Style.STRUCT.FIELD)

#define CHECK_PARSE_NESTED_BOOL(STRUCT, FIELD)                                 \
  CHECK_PARSE_NESTED_BOOL_FIELD(STRUCT, FIELD, #FIELD)

#define CHECK_PARSE(TEXT, FIELD, VALUE)                                        \
  EXPECT_NE(VALUE, Style.FIELD) << "Initial value already the same!";          \
  EXPECT_EQ(0, parseConfiguration(TEXT, &Style).value());                      \
  EXPECT_EQ(VALUE, Style.FIELD) << "Unexpected value after parsing!"

#define CHECK_PARSE_NESTED_VALUE(TEXT, STRUCT, FIELD, VALUE)                   \
  EXPECT_NE(VALUE, Style.STRUCT.FIELD) << "Initial value already the same!";   \
  EXPECT_EQ(0, parseConfiguration(#STRUCT ":\n  " TEXT, &Style).value());      \
  EXPECT_EQ(VALUE, Style.STRUCT.FIELD) << "Unexpected value after parsing!"

TEST(ConfigParseTest, ParsesConfigurationBools) {
  FormatStyle Style = {};
  Style.Language = FormatStyle::LK_Cpp;
  CHECK_PARSE_BOOL(AllowAllArgumentsOnNextLine);
  CHECK_PARSE_BOOL(AllowAllParametersOfDeclarationOnNextLine);
  CHECK_PARSE_BOOL(AllowShortCaseLabelsOnASingleLine);
  CHECK_PARSE_BOOL(AllowShortCompoundRequirementOnASingleLine);
  CHECK_PARSE_BOOL(AllowShortEnumsOnASingleLine);
  CHECK_PARSE_BOOL(AllowShortLoopsOnASingleLine);
  CHECK_PARSE_BOOL(BinPackArguments);
  CHECK_PARSE_BOOL(BinPackParameters);
  CHECK_PARSE_BOOL(BreakAdjacentStringLiterals);
  CHECK_PARSE_BOOL(BreakAfterJavaFieldAnnotations);
  CHECK_PARSE_BOOL(BreakBeforeTernaryOperators);
  CHECK_PARSE_BOOL(BreakStringLiterals);
  CHECK_PARSE_BOOL(CompactNamespaces);
  CHECK_PARSE_BOOL(DerivePointerAlignment);
  CHECK_PARSE_BOOL_FIELD(DerivePointerAlignment, "DerivePointerBinding");
  CHECK_PARSE_BOOL(DisableFormat);
  CHECK_PARSE_BOOL(IndentAccessModifiers);
  CHECK_PARSE_BOOL(IndentCaseLabels);
  CHECK_PARSE_BOOL(IndentCaseBlocks);
  CHECK_PARSE_BOOL(IndentGotoLabels);
  CHECK_PARSE_BOOL_FIELD(IndentRequiresClause, "IndentRequires");
  CHECK_PARSE_BOOL(IndentRequiresClause);
  CHECK_PARSE_BOOL(IndentWrappedFunctionNames);
  CHECK_PARSE_BOOL(InsertBraces);
  CHECK_PARSE_BOOL(InsertNewlineAtEOF);
  CHECK_PARSE_BOOL(KeepEmptyLinesAtEOF);
  CHECK_PARSE_BOOL(KeepEmptyLinesAtTheStartOfBlocks);
  CHECK_PARSE_BOOL(ObjCSpaceAfterProperty);
  CHECK_PARSE_BOOL(ObjCSpaceBeforeProtocolList);
  CHECK_PARSE_BOOL(Cpp11BracedListStyle);
  CHECK_PARSE_BOOL(ReflowComments);
  CHECK_PARSE_BOOL(RemoveBracesLLVM);
  CHECK_PARSE_BOOL(RemoveSemicolon);
  CHECK_PARSE_BOOL(SkipMacroDefinitionBody);
  CHECK_PARSE_BOOL(SpacesInSquareBrackets);
  CHECK_PARSE_BOOL(SpaceInEmptyBlock);
  CHECK_PARSE_BOOL(SpacesInContainerLiterals);
  CHECK_PARSE_BOOL(SpaceAfterCStyleCast);
  CHECK_PARSE_BOOL(SpaceAfterTemplateKeyword);
  CHECK_PARSE_BOOL(SpaceAfterLogicalNot);
  CHECK_PARSE_BOOL(SpaceBeforeAssignmentOperators);
  CHECK_PARSE_BOOL(SpaceBeforeCaseColon);
  CHECK_PARSE_BOOL(SpaceBeforeCpp11BracedList);
  CHECK_PARSE_BOOL(SpaceBeforeCtorInitializerColon);
  CHECK_PARSE_BOOL(SpaceBeforeInheritanceColon);
  CHECK_PARSE_BOOL(SpaceBeforeJsonColon);
  CHECK_PARSE_BOOL(SpaceBeforeRangeBasedForLoopColon);
  CHECK_PARSE_BOOL(SpaceBeforeSquareBrackets);
  CHECK_PARSE_BOOL(VerilogBreakBetweenInstancePorts);

  CHECK_PARSE_NESTED_BOOL(AlignConsecutiveShortCaseStatements, Enabled);
  CHECK_PARSE_NESTED_BOOL(AlignConsecutiveShortCaseStatements,
                          AcrossEmptyLines);
  CHECK_PARSE_NESTED_BOOL(AlignConsecutiveShortCaseStatements, AcrossComments);
  CHECK_PARSE_NESTED_BOOL(AlignConsecutiveShortCaseStatements, AlignCaseColons);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, AfterCaseLabel);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, AfterClass);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, AfterEnum);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, AfterFunction);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, AfterNamespace);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, AfterObjCDeclaration);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, AfterStruct);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, AfterUnion);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, AfterExternBlock);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, BeforeCatch);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, BeforeElse);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, BeforeLambdaBody);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, BeforeWhile);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, IndentBraces);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, SplitEmptyFunction);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, SplitEmptyRecord);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, SplitEmptyNamespace);
  CHECK_PARSE_NESTED_BOOL(SpaceBeforeParensOptions, AfterControlStatements);
  CHECK_PARSE_NESTED_BOOL(SpaceBeforeParensOptions, AfterForeachMacros);
  CHECK_PARSE_NESTED_BOOL(SpaceBeforeParensOptions,
                          AfterFunctionDeclarationName);
  CHECK_PARSE_NESTED_BOOL(SpaceBeforeParensOptions,
                          AfterFunctionDefinitionName);
  CHECK_PARSE_NESTED_BOOL(SpaceBeforeParensOptions, AfterIfMacros);
  CHECK_PARSE_NESTED_BOOL(SpaceBeforeParensOptions, AfterOverloadedOperator);
  CHECK_PARSE_NESTED_BOOL(SpaceBeforeParensOptions, BeforeNonEmptyParentheses);
  CHECK_PARSE_NESTED_BOOL(SpacesInParensOptions, InCStyleCasts);
  CHECK_PARSE_NESTED_BOOL(SpacesInParensOptions, InConditionalStatements);
  CHECK_PARSE_NESTED_BOOL(SpacesInParensOptions, InEmptyParentheses);
  CHECK_PARSE_NESTED_BOOL(SpacesInParensOptions, Other);
}

#undef CHECK_PARSE_BOOL

TEST(ConfigParseTest, ParsesConfiguration) {
  FormatStyle Style = {};
  Style.Language = FormatStyle::LK_Cpp;
  CHECK_PARSE("AccessModifierOffset: -1234", AccessModifierOffset, -1234);
  CHECK_PARSE("ConstructorInitializerIndentWidth: 1234",
              ConstructorInitializerIndentWidth, 1234u);
  CHECK_PARSE("ObjCBlockIndentWidth: 1234", ObjCBlockIndentWidth, 1234u);
  CHECK_PARSE("ColumnLimit: 1234", ColumnLimit, 1234u);
  CHECK_PARSE("MaxEmptyLinesToKeep: 1234", MaxEmptyLinesToKeep, 1234u);
  CHECK_PARSE("PenaltyBreakAssignment: 1234", PenaltyBreakAssignment, 1234u);
  CHECK_PARSE("PenaltyBreakBeforeFirstCallParameter: 1234",
              PenaltyBreakBeforeFirstCallParameter, 1234u);
  CHECK_PARSE("PenaltyBreakTemplateDeclaration: 1234",
              PenaltyBreakTemplateDeclaration, 1234u);
  CHECK_PARSE("PenaltyBreakOpenParenthesis: 1234", PenaltyBreakOpenParenthesis,
              1234u);
  CHECK_PARSE("PenaltyBreakScopeResolution: 1234", PenaltyBreakScopeResolution,
              1234u);
  CHECK_PARSE("PenaltyExcessCharacter: 1234", PenaltyExcessCharacter, 1234u);
  CHECK_PARSE("PenaltyReturnTypeOnItsOwnLine: 1234",
              PenaltyReturnTypeOnItsOwnLine, 1234u);
  CHECK_PARSE("SpacesBeforeTrailingComments: 1234",
              SpacesBeforeTrailingComments, 1234u);
  CHECK_PARSE("IndentWidth: 32", IndentWidth, 32u);
  CHECK_PARSE("ContinuationIndentWidth: 11", ContinuationIndentWidth, 11u);
  CHECK_PARSE("BracedInitializerIndentWidth: 34", BracedInitializerIndentWidth,
              34);
  CHECK_PARSE("CommentPragmas: '// abc$'", CommentPragmas, "// abc$");

  Style.QualifierAlignment = FormatStyle::QAS_Right;
  CHECK_PARSE("QualifierAlignment: Leave", QualifierAlignment,
              FormatStyle::QAS_Leave);
  CHECK_PARSE("QualifierAlignment: Right", QualifierAlignment,
              FormatStyle::QAS_Right);
  CHECK_PARSE("QualifierAlignment: Left", QualifierAlignment,
              FormatStyle::QAS_Left);
  CHECK_PARSE("QualifierAlignment: Custom", QualifierAlignment,
              FormatStyle::QAS_Custom);

  Style.QualifierOrder.clear();
  CHECK_PARSE("QualifierOrder: [ const, volatile, type ]", QualifierOrder,
              std::vector<std::string>({"const", "volatile", "type"}));
  Style.QualifierOrder.clear();
  CHECK_PARSE("QualifierOrder: [const, type]", QualifierOrder,
              std::vector<std::string>({"const", "type"}));
  Style.QualifierOrder.clear();
  CHECK_PARSE("QualifierOrder: [volatile, type]", QualifierOrder,
              std::vector<std::string>({"volatile", "type"}));

#define CHECK_ALIGN_CONSECUTIVE(FIELD)                                         \
  do {                                                                         \
    Style.FIELD.Enabled = true;                                                \
    CHECK_PARSE(                                                               \
        #FIELD ": None", FIELD,                                                \
        FormatStyle::AlignConsecutiveStyle(                                    \
            {/*Enabled=*/false, /*AcrossEmptyLines=*/false,                    \
             /*AcrossComments=*/false, /*AlignCompound=*/false,                \
             /*AlignFunctionPointers=*/false, /*PadOperators=*/true}));        \
    CHECK_PARSE(                                                               \
        #FIELD ": Consecutive", FIELD,                                         \
        FormatStyle::AlignConsecutiveStyle(                                    \
            {/*Enabled=*/true, /*AcrossEmptyLines=*/false,                     \
             /*AcrossComments=*/false, /*AlignCompound=*/false,                \
             /*AlignFunctionPointers=*/false, /*PadOperators=*/true}));        \
    CHECK_PARSE(                                                               \
        #FIELD ": AcrossEmptyLines", FIELD,                                    \
        FormatStyle::AlignConsecutiveStyle(                                    \
            {/*Enabled=*/true, /*AcrossEmptyLines=*/true,                      \
             /*AcrossComments=*/false, /*AlignCompound=*/false,                \
             /*AlignFunctionPointers=*/false, /*PadOperators=*/true}));        \
    CHECK_PARSE(                                                               \
        #FIELD ": AcrossEmptyLinesAndComments", FIELD,                         \
        FormatStyle::AlignConsecutiveStyle(                                    \
            {/*Enabled=*/true, /*AcrossEmptyLines=*/true,                      \
             /*AcrossComments=*/true, /*AlignCompound=*/false,                 \
             /*AlignFunctionPointers=*/false, /*PadOperators=*/true}));        \
    /* For backwards compability, false / true should still parse */           \
    CHECK_PARSE(                                                               \
        #FIELD ": false", FIELD,                                               \
        FormatStyle::AlignConsecutiveStyle(                                    \
            {/*Enabled=*/false, /*AcrossEmptyLines=*/false,                    \
             /*AcrossComments=*/false, /*AlignCompound=*/false,                \
             /*AlignFunctionPointers=*/false, /*PadOperators=*/true}));        \
    CHECK_PARSE(                                                               \
        #FIELD ": true", FIELD,                                                \
        FormatStyle::AlignConsecutiveStyle(                                    \
            {/*Enabled=*/true, /*AcrossEmptyLines=*/false,                     \
             /*AcrossComments=*/false, /*AlignCompound=*/false,                \
             /*AlignFunctionPointers=*/false, /*PadOperators=*/true}));        \
                                                                               \
    CHECK_PARSE_NESTED_BOOL(FIELD, Enabled);                                   \
    CHECK_PARSE_NESTED_BOOL(FIELD, AcrossEmptyLines);                          \
    CHECK_PARSE_NESTED_BOOL(FIELD, AcrossComments);                            \
    CHECK_PARSE_NESTED_BOOL(FIELD, AlignCompound);                             \
    CHECK_PARSE_NESTED_BOOL(FIELD, PadOperators);                              \
  } while (false)

  CHECK_ALIGN_CONSECUTIVE(AlignConsecutiveAssignments);
  CHECK_ALIGN_CONSECUTIVE(AlignConsecutiveBitFields);
  CHECK_ALIGN_CONSECUTIVE(AlignConsecutiveMacros);
  CHECK_ALIGN_CONSECUTIVE(AlignConsecutiveDeclarations);

#undef CHECK_ALIGN_CONSECUTIVE

  Style.PointerAlignment = FormatStyle::PAS_Middle;
  CHECK_PARSE("PointerAlignment: Left", PointerAlignment,
              FormatStyle::PAS_Left);
  CHECK_PARSE("PointerAlignment: Right", PointerAlignment,
              FormatStyle::PAS_Right);
  CHECK_PARSE("PointerAlignment: Middle", PointerAlignment,
              FormatStyle::PAS_Middle);
  Style.ReferenceAlignment = FormatStyle::RAS_Middle;
  CHECK_PARSE("ReferenceAlignment: Pointer", ReferenceAlignment,
              FormatStyle::RAS_Pointer);
  CHECK_PARSE("ReferenceAlignment: Left", ReferenceAlignment,
              FormatStyle::RAS_Left);
  CHECK_PARSE("ReferenceAlignment: Right", ReferenceAlignment,
              FormatStyle::RAS_Right);
  CHECK_PARSE("ReferenceAlignment: Middle", ReferenceAlignment,
              FormatStyle::RAS_Middle);
  // For backward compatibility:
  CHECK_PARSE("PointerBindsToType: Left", PointerAlignment,
              FormatStyle::PAS_Left);
  CHECK_PARSE("PointerBindsToType: Right", PointerAlignment,
              FormatStyle::PAS_Right);
  CHECK_PARSE("PointerBindsToType: Middle", PointerAlignment,
              FormatStyle::PAS_Middle);

  Style.Standard = FormatStyle::LS_Auto;
  CHECK_PARSE("Standard: c++03", Standard, FormatStyle::LS_Cpp03);
  CHECK_PARSE("Standard: c++11", Standard, FormatStyle::LS_Cpp11);
  CHECK_PARSE("Standard: c++14", Standard, FormatStyle::LS_Cpp14);
  CHECK_PARSE("Standard: c++17", Standard, FormatStyle::LS_Cpp17);
  CHECK_PARSE("Standard: c++20", Standard, FormatStyle::LS_Cpp20);
  CHECK_PARSE("Standard: Auto", Standard, FormatStyle::LS_Auto);
  CHECK_PARSE("Standard: Latest", Standard, FormatStyle::LS_Latest);
  // Legacy aliases:
  CHECK_PARSE("Standard: Cpp03", Standard, FormatStyle::LS_Cpp03);
  CHECK_PARSE("Standard: Cpp11", Standard, FormatStyle::LS_Latest);
  CHECK_PARSE("Standard: C++03", Standard, FormatStyle::LS_Cpp03);
  CHECK_PARSE("Standard: C++11", Standard, FormatStyle::LS_Cpp11);

  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_All;
  CHECK_PARSE("BreakBeforeBinaryOperators: NonAssignment",
              BreakBeforeBinaryOperators, FormatStyle::BOS_NonAssignment);
  CHECK_PARSE("BreakBeforeBinaryOperators: None", BreakBeforeBinaryOperators,
              FormatStyle::BOS_None);
  CHECK_PARSE("BreakBeforeBinaryOperators: All", BreakBeforeBinaryOperators,
              FormatStyle::BOS_All);
  // For backward compatibility:
  CHECK_PARSE("BreakBeforeBinaryOperators: false", BreakBeforeBinaryOperators,
              FormatStyle::BOS_None);
  CHECK_PARSE("BreakBeforeBinaryOperators: true", BreakBeforeBinaryOperators,
              FormatStyle::BOS_All);

  Style.BreakConstructorInitializers = FormatStyle::BCIS_BeforeColon;
  CHECK_PARSE("BreakConstructorInitializers: BeforeComma",
              BreakConstructorInitializers, FormatStyle::BCIS_BeforeComma);
  CHECK_PARSE("BreakConstructorInitializers: AfterColon",
              BreakConstructorInitializers, FormatStyle::BCIS_AfterColon);
  CHECK_PARSE("BreakConstructorInitializers: BeforeColon",
              BreakConstructorInitializers, FormatStyle::BCIS_BeforeColon);
  // For backward compatibility:
  CHECK_PARSE("BreakConstructorInitializersBeforeComma: true",
              BreakConstructorInitializers, FormatStyle::BCIS_BeforeComma);

  Style.BreakInheritanceList = FormatStyle::BILS_BeforeColon;
  CHECK_PARSE("BreakInheritanceList: AfterComma", BreakInheritanceList,
              FormatStyle::BILS_AfterComma);
  CHECK_PARSE("BreakInheritanceList: BeforeComma", BreakInheritanceList,
              FormatStyle::BILS_BeforeComma);
  CHECK_PARSE("BreakInheritanceList: AfterColon", BreakInheritanceList,
              FormatStyle::BILS_AfterColon);
  CHECK_PARSE("BreakInheritanceList: BeforeColon", BreakInheritanceList,
              FormatStyle::BILS_BeforeColon);
  // For backward compatibility:
  CHECK_PARSE("BreakBeforeInheritanceComma: true", BreakInheritanceList,
              FormatStyle::BILS_BeforeComma);

  Style.PackConstructorInitializers = FormatStyle::PCIS_BinPack;
  CHECK_PARSE("PackConstructorInitializers: Never", PackConstructorInitializers,
              FormatStyle::PCIS_Never);
  CHECK_PARSE("PackConstructorInitializers: BinPack",
              PackConstructorInitializers, FormatStyle::PCIS_BinPack);
  CHECK_PARSE("PackConstructorInitializers: CurrentLine",
              PackConstructorInitializers, FormatStyle::PCIS_CurrentLine);
  CHECK_PARSE("PackConstructorInitializers: NextLine",
              PackConstructorInitializers, FormatStyle::PCIS_NextLine);
  CHECK_PARSE("PackConstructorInitializers: NextLineOnly",
              PackConstructorInitializers, FormatStyle::PCIS_NextLineOnly);
  // For backward compatibility:
  CHECK_PARSE("BasedOnStyle: Google\n"
              "ConstructorInitializerAllOnOneLineOrOnePerLine: true\n"
              "AllowAllConstructorInitializersOnNextLine: false",
              PackConstructorInitializers, FormatStyle::PCIS_CurrentLine);
  Style.PackConstructorInitializers = FormatStyle::PCIS_NextLine;
  CHECK_PARSE("BasedOnStyle: Google\n"
              "ConstructorInitializerAllOnOneLineOrOnePerLine: false",
              PackConstructorInitializers, FormatStyle::PCIS_BinPack);
  CHECK_PARSE("ConstructorInitializerAllOnOneLineOrOnePerLine: true\n"
              "AllowAllConstructorInitializersOnNextLine: true",
              PackConstructorInitializers, FormatStyle::PCIS_NextLine);
  Style.PackConstructorInitializers = FormatStyle::PCIS_BinPack;
  CHECK_PARSE("ConstructorInitializerAllOnOneLineOrOnePerLine: true\n"
              "AllowAllConstructorInitializersOnNextLine: false",
              PackConstructorInitializers, FormatStyle::PCIS_CurrentLine);

  Style.EmptyLineBeforeAccessModifier = FormatStyle::ELBAMS_LogicalBlock;
  CHECK_PARSE("EmptyLineBeforeAccessModifier: Never",
              EmptyLineBeforeAccessModifier, FormatStyle::ELBAMS_Never);
  CHECK_PARSE("EmptyLineBeforeAccessModifier: Leave",
              EmptyLineBeforeAccessModifier, FormatStyle::ELBAMS_Leave);
  CHECK_PARSE("EmptyLineBeforeAccessModifier: LogicalBlock",
              EmptyLineBeforeAccessModifier, FormatStyle::ELBAMS_LogicalBlock);
  CHECK_PARSE("EmptyLineBeforeAccessModifier: Always",
              EmptyLineBeforeAccessModifier, FormatStyle::ELBAMS_Always);

  Style.AlignAfterOpenBracket = FormatStyle::BAS_AlwaysBreak;
  CHECK_PARSE("AlignAfterOpenBracket: Align", AlignAfterOpenBracket,
              FormatStyle::BAS_Align);
  CHECK_PARSE("AlignAfterOpenBracket: DontAlign", AlignAfterOpenBracket,
              FormatStyle::BAS_DontAlign);
  CHECK_PARSE("AlignAfterOpenBracket: AlwaysBreak", AlignAfterOpenBracket,
              FormatStyle::BAS_AlwaysBreak);
  CHECK_PARSE("AlignAfterOpenBracket: BlockIndent", AlignAfterOpenBracket,
              FormatStyle::BAS_BlockIndent);
  // For backward compatibility:
  CHECK_PARSE("AlignAfterOpenBracket: false", AlignAfterOpenBracket,
              FormatStyle::BAS_DontAlign);
  CHECK_PARSE("AlignAfterOpenBracket: true", AlignAfterOpenBracket,
              FormatStyle::BAS_Align);

  Style.AlignEscapedNewlines = FormatStyle::ENAS_Left;
  CHECK_PARSE("AlignEscapedNewlines: DontAlign", AlignEscapedNewlines,
              FormatStyle::ENAS_DontAlign);
  CHECK_PARSE("AlignEscapedNewlines: Left", AlignEscapedNewlines,
              FormatStyle::ENAS_Left);
  CHECK_PARSE("AlignEscapedNewlines: Right", AlignEscapedNewlines,
              FormatStyle::ENAS_Right);
  // For backward compatibility:
  CHECK_PARSE("AlignEscapedNewlinesLeft: true", AlignEscapedNewlines,
              FormatStyle::ENAS_Left);
  CHECK_PARSE("AlignEscapedNewlinesLeft: false", AlignEscapedNewlines,
              FormatStyle::ENAS_Right);

  Style.AlignOperands = FormatStyle::OAS_Align;
  CHECK_PARSE("AlignOperands: DontAlign", AlignOperands,
              FormatStyle::OAS_DontAlign);
  CHECK_PARSE("AlignOperands: Align", AlignOperands, FormatStyle::OAS_Align);
  CHECK_PARSE("AlignOperands: AlignAfterOperator", AlignOperands,
              FormatStyle::OAS_AlignAfterOperator);
  // For backward compatibility:
  CHECK_PARSE("AlignOperands: false", AlignOperands,
              FormatStyle::OAS_DontAlign);
  CHECK_PARSE("AlignOperands: true", AlignOperands, FormatStyle::OAS_Align);

  CHECK_PARSE("AlignTrailingComments: Leave", AlignTrailingComments,
              FormatStyle::TrailingCommentsAlignmentStyle(
                  {FormatStyle::TCAS_Leave, 0}));
  CHECK_PARSE("AlignTrailingComments: Always", AlignTrailingComments,
              FormatStyle::TrailingCommentsAlignmentStyle(
                  {FormatStyle::TCAS_Always, 0}));
  CHECK_PARSE("AlignTrailingComments: Never", AlignTrailingComments,
              FormatStyle::TrailingCommentsAlignmentStyle(
                  {FormatStyle::TCAS_Never, 0}));
  // For backwards compatibility
  CHECK_PARSE("AlignTrailingComments: true", AlignTrailingComments,
              FormatStyle::TrailingCommentsAlignmentStyle(
                  {FormatStyle::TCAS_Always, 0}));
  CHECK_PARSE("AlignTrailingComments: false", AlignTrailingComments,
              FormatStyle::TrailingCommentsAlignmentStyle(
                  {FormatStyle::TCAS_Never, 0}));
  CHECK_PARSE_NESTED_VALUE("Kind: Always", AlignTrailingComments, Kind,
                           FormatStyle::TCAS_Always);
  CHECK_PARSE_NESTED_VALUE("Kind: Never", AlignTrailingComments, Kind,
                           FormatStyle::TCAS_Never);
  CHECK_PARSE_NESTED_VALUE("Kind: Leave", AlignTrailingComments, Kind,
                           FormatStyle::TCAS_Leave);
  CHECK_PARSE_NESTED_VALUE("OverEmptyLines: 1234", AlignTrailingComments,
                           OverEmptyLines, 1234u);

  Style.UseTab = FormatStyle::UT_ForIndentation;
  CHECK_PARSE("UseTab: Never", UseTab, FormatStyle::UT_Never);
  CHECK_PARSE("UseTab: ForIndentation", UseTab, FormatStyle::UT_ForIndentation);
  CHECK_PARSE("UseTab: Always", UseTab, FormatStyle::UT_Always);
  CHECK_PARSE("UseTab: ForContinuationAndIndentation", UseTab,
              FormatStyle::UT_ForContinuationAndIndentation);
  CHECK_PARSE("UseTab: AlignWithSpaces", UseTab,
              FormatStyle::UT_AlignWithSpaces);
  // For backward compatibility:
  CHECK_PARSE("UseTab: false", UseTab, FormatStyle::UT_Never);
  CHECK_PARSE("UseTab: true", UseTab, FormatStyle::UT_Always);

  Style.AllowShortBlocksOnASingleLine = FormatStyle::SBS_Empty;
  CHECK_PARSE("AllowShortBlocksOnASingleLine: Never",
              AllowShortBlocksOnASingleLine, FormatStyle::SBS_Never);
  CHECK_PARSE("AllowShortBlocksOnASingleLine: Empty",
              AllowShortBlocksOnASingleLine, FormatStyle::SBS_Empty);
  CHECK_PARSE("AllowShortBlocksOnASingleLine: Always",
              AllowShortBlocksOnASingleLine, FormatStyle::SBS_Always);
  // For backward compatibility:
  CHECK_PARSE("AllowShortBlocksOnASingleLine: false",
              AllowShortBlocksOnASingleLine, FormatStyle::SBS_Never);
  CHECK_PARSE("AllowShortBlocksOnASingleLine: true",
              AllowShortBlocksOnASingleLine, FormatStyle::SBS_Always);

  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_Inline;
  CHECK_PARSE("AllowShortFunctionsOnASingleLine: None",
              AllowShortFunctionsOnASingleLine, FormatStyle::SFS_None);
  CHECK_PARSE("AllowShortFunctionsOnASingleLine: Inline",
              AllowShortFunctionsOnASingleLine, FormatStyle::SFS_Inline);
  CHECK_PARSE("AllowShortFunctionsOnASingleLine: Empty",
              AllowShortFunctionsOnASingleLine, FormatStyle::SFS_Empty);
  CHECK_PARSE("AllowShortFunctionsOnASingleLine: All",
              AllowShortFunctionsOnASingleLine, FormatStyle::SFS_All);
  // For backward compatibility:
  CHECK_PARSE("AllowShortFunctionsOnASingleLine: false",
              AllowShortFunctionsOnASingleLine, FormatStyle::SFS_None);
  CHECK_PARSE("AllowShortFunctionsOnASingleLine: true",
              AllowShortFunctionsOnASingleLine, FormatStyle::SFS_All);

  Style.AllowShortLambdasOnASingleLine = FormatStyle::SLS_All;
  CHECK_PARSE("AllowShortLambdasOnASingleLine: None",
              AllowShortLambdasOnASingleLine, FormatStyle::SLS_None);
  CHECK_PARSE("AllowShortLambdasOnASingleLine: Empty",
              AllowShortLambdasOnASingleLine, FormatStyle::SLS_Empty);
  CHECK_PARSE("AllowShortLambdasOnASingleLine: Inline",
              AllowShortLambdasOnASingleLine, FormatStyle::SLS_Inline);
  CHECK_PARSE("AllowShortLambdasOnASingleLine: All",
              AllowShortLambdasOnASingleLine, FormatStyle::SLS_All);
  // For backward compatibility:
  CHECK_PARSE("AllowShortLambdasOnASingleLine: false",
              AllowShortLambdasOnASingleLine, FormatStyle::SLS_None);
  CHECK_PARSE("AllowShortLambdasOnASingleLine: true",
              AllowShortLambdasOnASingleLine, FormatStyle::SLS_All);

  Style.SpaceAroundPointerQualifiers = FormatStyle::SAPQ_Both;
  CHECK_PARSE("SpaceAroundPointerQualifiers: Default",
              SpaceAroundPointerQualifiers, FormatStyle::SAPQ_Default);
  CHECK_PARSE("SpaceAroundPointerQualifiers: Before",
              SpaceAroundPointerQualifiers, FormatStyle::SAPQ_Before);
  CHECK_PARSE("SpaceAroundPointerQualifiers: After",
              SpaceAroundPointerQualifiers, FormatStyle::SAPQ_After);
  CHECK_PARSE("SpaceAroundPointerQualifiers: Both",
              SpaceAroundPointerQualifiers, FormatStyle::SAPQ_Both);

  Style.SpaceBeforeParens = FormatStyle::SBPO_Always;
  CHECK_PARSE("SpaceBeforeParens: Never", SpaceBeforeParens,
              FormatStyle::SBPO_Never);
  CHECK_PARSE("SpaceBeforeParens: Always", SpaceBeforeParens,
              FormatStyle::SBPO_Always);
  CHECK_PARSE("SpaceBeforeParens: ControlStatements", SpaceBeforeParens,
              FormatStyle::SBPO_ControlStatements);
  CHECK_PARSE("SpaceBeforeParens: ControlStatementsExceptControlMacros",
              SpaceBeforeParens,
              FormatStyle::SBPO_ControlStatementsExceptControlMacros);
  CHECK_PARSE("SpaceBeforeParens: NonEmptyParentheses", SpaceBeforeParens,
              FormatStyle::SBPO_NonEmptyParentheses);
  CHECK_PARSE("SpaceBeforeParens: Custom", SpaceBeforeParens,
              FormatStyle::SBPO_Custom);
  // For backward compatibility:
  CHECK_PARSE("SpaceAfterControlStatementKeyword: false", SpaceBeforeParens,
              FormatStyle::SBPO_Never);
  CHECK_PARSE("SpaceAfterControlStatementKeyword: true", SpaceBeforeParens,
              FormatStyle::SBPO_ControlStatements);
  CHECK_PARSE("SpaceBeforeParens: ControlStatementsExceptForEachMacros",
              SpaceBeforeParens,
              FormatStyle::SBPO_ControlStatementsExceptControlMacros);

  Style.SpaceBeforeParens = FormatStyle::SBPO_Custom;
  Style.SpaceBeforeParensOptions.AfterPlacementOperator =
      FormatStyle::SpaceBeforeParensCustom::APO_Always;
  CHECK_PARSE("SpaceBeforeParensOptions:\n"
              "  AfterPlacementOperator: Never",
              SpaceBeforeParensOptions.AfterPlacementOperator,
              FormatStyle::SpaceBeforeParensCustom::APO_Never);

  CHECK_PARSE("SpaceBeforeParensOptions:\n"
              "  AfterPlacementOperator: Always",
              SpaceBeforeParensOptions.AfterPlacementOperator,
              FormatStyle::SpaceBeforeParensCustom::APO_Always);

  CHECK_PARSE("SpaceBeforeParensOptions:\n"
              "  AfterPlacementOperator: Leave",
              SpaceBeforeParensOptions.AfterPlacementOperator,
              FormatStyle::SpaceBeforeParensCustom::APO_Leave);

  // For backward compatibility:
  Style.SpacesInParens = FormatStyle::SIPO_Never;
  Style.SpacesInParensOptions = {};
  CHECK_PARSE("SpacesInParentheses: true", SpacesInParens,
              FormatStyle::SIPO_Custom);
  Style.SpacesInParens = FormatStyle::SIPO_Never;
  Style.SpacesInParensOptions = {};
  CHECK_PARSE("SpacesInParentheses: true", SpacesInParensOptions,
              FormatStyle::SpacesInParensCustom(true, false, false, true));
  Style.SpacesInParens = FormatStyle::SIPO_Never;
  Style.SpacesInParensOptions = {};
  CHECK_PARSE("SpacesInConditionalStatement: true", SpacesInParensOptions,
              FormatStyle::SpacesInParensCustom(true, false, false, false));
  Style.SpacesInParens = FormatStyle::SIPO_Never;
  Style.SpacesInParensOptions = {};
  CHECK_PARSE("SpacesInCStyleCastParentheses: true", SpacesInParensOptions,
              FormatStyle::SpacesInParensCustom(false, true, false, false));
  Style.SpacesInParens = FormatStyle::SIPO_Never;
  Style.SpacesInParensOptions = {};
  CHECK_PARSE("SpaceInEmptyParentheses: true", SpacesInParensOptions,
              FormatStyle::SpacesInParensCustom(false, false, true, false));
  Style.SpacesInParens = FormatStyle::SIPO_Never;
  Style.SpacesInParensOptions = {};

  Style.ColumnLimit = 123;
  FormatStyle BaseStyle = getLLVMStyle();
  CHECK_PARSE("BasedOnStyle: LLVM", ColumnLimit, BaseStyle.ColumnLimit);
  CHECK_PARSE("BasedOnStyle: LLVM\nColumnLimit: 1234", ColumnLimit, 1234u);

  Style.BreakBeforeBraces = FormatStyle::BS_Stroustrup;
  CHECK_PARSE("BreakBeforeBraces: Attach", BreakBeforeBraces,
              FormatStyle::BS_Attach);
  CHECK_PARSE("BreakBeforeBraces: Linux", BreakBeforeBraces,
              FormatStyle::BS_Linux);
  CHECK_PARSE("BreakBeforeBraces: Mozilla", BreakBeforeBraces,
              FormatStyle::BS_Mozilla);
  CHECK_PARSE("BreakBeforeBraces: Stroustrup", BreakBeforeBraces,
              FormatStyle::BS_Stroustrup);
  CHECK_PARSE("BreakBeforeBraces: Allman", BreakBeforeBraces,
              FormatStyle::BS_Allman);
  CHECK_PARSE("BreakBeforeBraces: Whitesmiths", BreakBeforeBraces,
              FormatStyle::BS_Whitesmiths);
  CHECK_PARSE("BreakBeforeBraces: GNU", BreakBeforeBraces, FormatStyle::BS_GNU);
  CHECK_PARSE("BreakBeforeBraces: WebKit", BreakBeforeBraces,
              FormatStyle::BS_WebKit);
  CHECK_PARSE("BreakBeforeBraces: Custom", BreakBeforeBraces,
              FormatStyle::BS_Custom);

  Style.BraceWrapping.AfterControlStatement = FormatStyle::BWACS_Never;
  CHECK_PARSE("BraceWrapping:\n"
              "  AfterControlStatement: MultiLine",
              BraceWrapping.AfterControlStatement,
              FormatStyle::BWACS_MultiLine);
  CHECK_PARSE("BraceWrapping:\n"
              "  AfterControlStatement: Always",
              BraceWrapping.AfterControlStatement, FormatStyle::BWACS_Always);
  CHECK_PARSE("BraceWrapping:\n"
              "  AfterControlStatement: Never",
              BraceWrapping.AfterControlStatement, FormatStyle::BWACS_Never);
  // For backward compatibility:
  CHECK_PARSE("BraceWrapping:\n"
              "  AfterControlStatement: true",
              BraceWrapping.AfterControlStatement, FormatStyle::BWACS_Always);
  CHECK_PARSE("BraceWrapping:\n"
              "  AfterControlStatement: false",
              BraceWrapping.AfterControlStatement, FormatStyle::BWACS_Never);

  Style.AlwaysBreakAfterReturnType = FormatStyle::RTBS_All;
  CHECK_PARSE("AlwaysBreakAfterReturnType: None", AlwaysBreakAfterReturnType,
              FormatStyle::RTBS_None);
  CHECK_PARSE("AlwaysBreakAfterReturnType: All", AlwaysBreakAfterReturnType,
              FormatStyle::RTBS_All);
  CHECK_PARSE("AlwaysBreakAfterReturnType: TopLevel",
              AlwaysBreakAfterReturnType, FormatStyle::RTBS_TopLevel);
  CHECK_PARSE("AlwaysBreakAfterReturnType: AllDefinitions",
              AlwaysBreakAfterReturnType, FormatStyle::RTBS_AllDefinitions);
  CHECK_PARSE("AlwaysBreakAfterReturnType: TopLevelDefinitions",
              AlwaysBreakAfterReturnType,
              FormatStyle::RTBS_TopLevelDefinitions);

  Style.AlwaysBreakTemplateDeclarations = FormatStyle::BTDS_Yes;
  CHECK_PARSE("AlwaysBreakTemplateDeclarations: No",
              AlwaysBreakTemplateDeclarations, FormatStyle::BTDS_No);
  CHECK_PARSE("AlwaysBreakTemplateDeclarations: MultiLine",
              AlwaysBreakTemplateDeclarations, FormatStyle::BTDS_MultiLine);
  CHECK_PARSE("AlwaysBreakTemplateDeclarations: Yes",
              AlwaysBreakTemplateDeclarations, FormatStyle::BTDS_Yes);
  CHECK_PARSE("AlwaysBreakTemplateDeclarations: false",
              AlwaysBreakTemplateDeclarations, FormatStyle::BTDS_MultiLine);
  CHECK_PARSE("AlwaysBreakTemplateDeclarations: true",
              AlwaysBreakTemplateDeclarations, FormatStyle::BTDS_Yes);

  Style.AlwaysBreakAfterDefinitionReturnType = FormatStyle::DRTBS_All;
  CHECK_PARSE("AlwaysBreakAfterDefinitionReturnType: None",
              AlwaysBreakAfterDefinitionReturnType, FormatStyle::DRTBS_None);
  CHECK_PARSE("AlwaysBreakAfterDefinitionReturnType: All",
              AlwaysBreakAfterDefinitionReturnType, FormatStyle::DRTBS_All);
  CHECK_PARSE("AlwaysBreakAfterDefinitionReturnType: TopLevel",
              AlwaysBreakAfterDefinitionReturnType,
              FormatStyle::DRTBS_TopLevel);

  Style.NamespaceIndentation = FormatStyle::NI_All;
  CHECK_PARSE("NamespaceIndentation: None", NamespaceIndentation,
              FormatStyle::NI_None);
  CHECK_PARSE("NamespaceIndentation: Inner", NamespaceIndentation,
              FormatStyle::NI_Inner);
  CHECK_PARSE("NamespaceIndentation: All", NamespaceIndentation,
              FormatStyle::NI_All);

  Style.AllowShortIfStatementsOnASingleLine = FormatStyle::SIS_OnlyFirstIf;
  CHECK_PARSE("AllowShortIfStatementsOnASingleLine: Never",
              AllowShortIfStatementsOnASingleLine, FormatStyle::SIS_Never);
  CHECK_PARSE("AllowShortIfStatementsOnASingleLine: WithoutElse",
              AllowShortIfStatementsOnASingleLine,
              FormatStyle::SIS_WithoutElse);
  CHECK_PARSE("AllowShortIfStatementsOnASingleLine: OnlyFirstIf",
              AllowShortIfStatementsOnASingleLine,
              FormatStyle::SIS_OnlyFirstIf);
  CHECK_PARSE("AllowShortIfStatementsOnASingleLine: AllIfsAndElse",
              AllowShortIfStatementsOnASingleLine,
              FormatStyle::SIS_AllIfsAndElse);
  CHECK_PARSE("AllowShortIfStatementsOnASingleLine: Always",
              AllowShortIfStatementsOnASingleLine,
              FormatStyle::SIS_OnlyFirstIf);
  CHECK_PARSE("AllowShortIfStatementsOnASingleLine: false",
              AllowShortIfStatementsOnASingleLine, FormatStyle::SIS_Never);
  CHECK_PARSE("AllowShortIfStatementsOnASingleLine: true",
              AllowShortIfStatementsOnASingleLine,
              FormatStyle::SIS_WithoutElse);

  Style.IndentExternBlock = FormatStyle::IEBS_NoIndent;
  CHECK_PARSE("IndentExternBlock: AfterExternBlock", IndentExternBlock,
              FormatStyle::IEBS_AfterExternBlock);
  CHECK_PARSE("IndentExternBlock: Indent", IndentExternBlock,
              FormatStyle::IEBS_Indent);
  CHECK_PARSE("IndentExternBlock: NoIndent", IndentExternBlock,
              FormatStyle::IEBS_NoIndent);
  CHECK_PARSE("IndentExternBlock: true", IndentExternBlock,
              FormatStyle::IEBS_Indent);
  CHECK_PARSE("IndentExternBlock: false", IndentExternBlock,
              FormatStyle::IEBS_NoIndent);

  Style.BitFieldColonSpacing = FormatStyle::BFCS_None;
  CHECK_PARSE("BitFieldColonSpacing: Both", BitFieldColonSpacing,
              FormatStyle::BFCS_Both);
  CHECK_PARSE("BitFieldColonSpacing: None", BitFieldColonSpacing,
              FormatStyle::BFCS_None);
  CHECK_PARSE("BitFieldColonSpacing: Before", BitFieldColonSpacing,
              FormatStyle::BFCS_Before);
  CHECK_PARSE("BitFieldColonSpacing: After", BitFieldColonSpacing,
              FormatStyle::BFCS_After);

  Style.SortJavaStaticImport = FormatStyle::SJSIO_Before;
  CHECK_PARSE("SortJavaStaticImport: After", SortJavaStaticImport,
              FormatStyle::SJSIO_After);
  CHECK_PARSE("SortJavaStaticImport: Before", SortJavaStaticImport,
              FormatStyle::SJSIO_Before);

  Style.SortUsingDeclarations = FormatStyle::SUD_LexicographicNumeric;
  CHECK_PARSE("SortUsingDeclarations: Never", SortUsingDeclarations,
              FormatStyle::SUD_Never);
  CHECK_PARSE("SortUsingDeclarations: Lexicographic", SortUsingDeclarations,
              FormatStyle::SUD_Lexicographic);
  CHECK_PARSE("SortUsingDeclarations: LexicographicNumeric",
              SortUsingDeclarations, FormatStyle::SUD_LexicographicNumeric);
  // For backward compatibility:
  CHECK_PARSE("SortUsingDeclarations: false", SortUsingDeclarations,
              FormatStyle::SUD_Never);
  CHECK_PARSE("SortUsingDeclarations: true", SortUsingDeclarations,
              FormatStyle::SUD_LexicographicNumeric);

  // FIXME: This is required because parsing a configuration simply overwrites
  // the first N elements of the list instead of resetting it.
  Style.ForEachMacros.clear();
  std::vector<std::string> BoostForeach;
  BoostForeach.push_back("BOOST_FOREACH");
  CHECK_PARSE("ForEachMacros: [BOOST_FOREACH]", ForEachMacros, BoostForeach);
  std::vector<std::string> BoostAndQForeach;
  BoostAndQForeach.push_back("BOOST_FOREACH");
  BoostAndQForeach.push_back("Q_FOREACH");
  CHECK_PARSE("ForEachMacros: [BOOST_FOREACH, Q_FOREACH]", ForEachMacros,
              BoostAndQForeach);

  Style.IfMacros.clear();
  std::vector<std::string> CustomIfs;
  CustomIfs.push_back("MYIF");
  CHECK_PARSE("IfMacros: [MYIF]", IfMacros, CustomIfs);

  Style.AttributeMacros.clear();
  CHECK_PARSE("BasedOnStyle: LLVM", AttributeMacros,
              std::vector<std::string>{"__capability"});
  CHECK_PARSE("AttributeMacros: [attr1, attr2]", AttributeMacros,
              std::vector<std::string>({"attr1", "attr2"}));

  Style.StatementAttributeLikeMacros.clear();
  CHECK_PARSE("StatementAttributeLikeMacros: [emit,Q_EMIT]",
              StatementAttributeLikeMacros,
              std::vector<std::string>({"emit", "Q_EMIT"}));

  Style.StatementMacros.clear();
  CHECK_PARSE("StatementMacros: [QUNUSED]", StatementMacros,
              std::vector<std::string>{"QUNUSED"});
  CHECK_PARSE("StatementMacros: [QUNUSED, QT_REQUIRE_VERSION]", StatementMacros,
              std::vector<std::string>({"QUNUSED", "QT_REQUIRE_VERSION"}));

  Style.NamespaceMacros.clear();
  CHECK_PARSE("NamespaceMacros: [TESTSUITE]", NamespaceMacros,
              std::vector<std::string>{"TESTSUITE"});
  CHECK_PARSE("NamespaceMacros: [TESTSUITE, SUITE]", NamespaceMacros,
              std::vector<std::string>({"TESTSUITE", "SUITE"}));

  Style.WhitespaceSensitiveMacros.clear();
  CHECK_PARSE("WhitespaceSensitiveMacros: [STRINGIZE]",
              WhitespaceSensitiveMacros, std::vector<std::string>{"STRINGIZE"});
  CHECK_PARSE("WhitespaceSensitiveMacros: [STRINGIZE, ASSERT]",
              WhitespaceSensitiveMacros,
              std::vector<std::string>({"STRINGIZE", "ASSERT"}));
  Style.WhitespaceSensitiveMacros.clear();
  CHECK_PARSE("WhitespaceSensitiveMacros: ['STRINGIZE']",
              WhitespaceSensitiveMacros, std::vector<std::string>{"STRINGIZE"});
  CHECK_PARSE("WhitespaceSensitiveMacros: ['STRINGIZE', 'ASSERT']",
              WhitespaceSensitiveMacros,
              std::vector<std::string>({"STRINGIZE", "ASSERT"}));

  Style.IncludeStyle.IncludeCategories.clear();
  std::vector<tooling::IncludeStyle::IncludeCategory> ExpectedCategories = {
      {"abc/.*", 2, 0, false}, {".*", 1, 0, true}};
  CHECK_PARSE("IncludeCategories:\n"
              "  - Regex: abc/.*\n"
              "    Priority: 2\n"
              "  - Regex: .*\n"
              "    Priority: 1\n"
              "    CaseSensitive: true",
              IncludeStyle.IncludeCategories, ExpectedCategories);
  CHECK_PARSE("IncludeIsMainRegex: 'abc$'", IncludeStyle.IncludeIsMainRegex,
              "abc$");
  CHECK_PARSE("IncludeIsMainSourceRegex: 'abc$'",
              IncludeStyle.IncludeIsMainSourceRegex, "abc$");

  Style.SortIncludes = FormatStyle::SI_Never;
  CHECK_PARSE("SortIncludes: true", SortIncludes,
              FormatStyle::SI_CaseSensitive);
  CHECK_PARSE("SortIncludes: false", SortIncludes, FormatStyle::SI_Never);
  CHECK_PARSE("SortIncludes: CaseInsensitive", SortIncludes,
              FormatStyle::SI_CaseInsensitive);
  CHECK_PARSE("SortIncludes: CaseSensitive", SortIncludes,
              FormatStyle::SI_CaseSensitive);
  CHECK_PARSE("SortIncludes: Never", SortIncludes, FormatStyle::SI_Never);

  Style.RawStringFormats.clear();
  std::vector<FormatStyle::RawStringFormat> ExpectedRawStringFormats = {
      {
          FormatStyle::LK_TextProto,
          {"pb", "proto"},
          {"PARSE_TEXT_PROTO"},
          /*CanonicalDelimiter=*/"",
          "llvm",
      },
      {
          FormatStyle::LK_Cpp,
          {"cc", "cpp"},
          {"C_CODEBLOCK", "CPPEVAL"},
          /*CanonicalDelimiter=*/"cc",
          /*BasedOnStyle=*/"",
      },
  };

  CHECK_PARSE("RawStringFormats:\n"
              "  - Language: TextProto\n"
              "    Delimiters:\n"
              "      - 'pb'\n"
              "      - 'proto'\n"
              "    EnclosingFunctions:\n"
              "      - 'PARSE_TEXT_PROTO'\n"
              "    BasedOnStyle: llvm\n"
              "  - Language: Cpp\n"
              "    Delimiters:\n"
              "      - 'cc'\n"
              "      - 'cpp'\n"
              "    EnclosingFunctions:\n"
              "      - 'C_CODEBLOCK'\n"
              "      - 'CPPEVAL'\n"
              "    CanonicalDelimiter: 'cc'",
              RawStringFormats, ExpectedRawStringFormats);

  CHECK_PARSE("SpacesInLineCommentPrefix:\n"
              "  Minimum: 0\n"
              "  Maximum: 0",
              SpacesInLineCommentPrefix.Minimum, 0u);
  EXPECT_EQ(Style.SpacesInLineCommentPrefix.Maximum, 0u);
  Style.SpacesInLineCommentPrefix.Minimum = 1;
  CHECK_PARSE("SpacesInLineCommentPrefix:\n"
              "  Minimum: 2",
              SpacesInLineCommentPrefix.Minimum, 0u);
  CHECK_PARSE("SpacesInLineCommentPrefix:\n"
              "  Maximum: -1",
              SpacesInLineCommentPrefix.Maximum, -1u);
  CHECK_PARSE("SpacesInLineCommentPrefix:\n"
              "  Minimum: 2",
              SpacesInLineCommentPrefix.Minimum, 2u);
  CHECK_PARSE("SpacesInLineCommentPrefix:\n"
              "  Maximum: 1",
              SpacesInLineCommentPrefix.Maximum, 1u);
  EXPECT_EQ(Style.SpacesInLineCommentPrefix.Minimum, 1u);

  Style.SpacesInAngles = FormatStyle::SIAS_Always;
  CHECK_PARSE("SpacesInAngles: Never", SpacesInAngles, FormatStyle::SIAS_Never);
  CHECK_PARSE("SpacesInAngles: Always", SpacesInAngles,
              FormatStyle::SIAS_Always);
  CHECK_PARSE("SpacesInAngles: Leave", SpacesInAngles, FormatStyle::SIAS_Leave);
  // For backward compatibility:
  CHECK_PARSE("SpacesInAngles: false", SpacesInAngles, FormatStyle::SIAS_Never);
  CHECK_PARSE("SpacesInAngles: true", SpacesInAngles, FormatStyle::SIAS_Always);

  CHECK_PARSE("RequiresClausePosition: WithPreceding", RequiresClausePosition,
              FormatStyle::RCPS_WithPreceding);
  CHECK_PARSE("RequiresClausePosition: WithFollowing", RequiresClausePosition,
              FormatStyle::RCPS_WithFollowing);
  CHECK_PARSE("RequiresClausePosition: SingleLine", RequiresClausePosition,
              FormatStyle::RCPS_SingleLine);
  CHECK_PARSE("RequiresClausePosition: OwnLine", RequiresClausePosition,
              FormatStyle::RCPS_OwnLine);

  CHECK_PARSE("BreakBeforeConceptDeclarations: Never",
              BreakBeforeConceptDeclarations, FormatStyle::BBCDS_Never);
  CHECK_PARSE("BreakBeforeConceptDeclarations: Always",
              BreakBeforeConceptDeclarations, FormatStyle::BBCDS_Always);
  CHECK_PARSE("BreakBeforeConceptDeclarations: Allowed",
              BreakBeforeConceptDeclarations, FormatStyle::BBCDS_Allowed);
  // For backward compatibility:
  CHECK_PARSE("BreakBeforeConceptDeclarations: true",
              BreakBeforeConceptDeclarations, FormatStyle::BBCDS_Always);
  CHECK_PARSE("BreakBeforeConceptDeclarations: false",
              BreakBeforeConceptDeclarations, FormatStyle::BBCDS_Allowed);

  CHECK_PARSE("BreakAfterAttributes: Always", BreakAfterAttributes,
              FormatStyle::ABS_Always);
  CHECK_PARSE("BreakAfterAttributes: Leave", BreakAfterAttributes,
              FormatStyle::ABS_Leave);
  CHECK_PARSE("BreakAfterAttributes: Never", BreakAfterAttributes,
              FormatStyle::ABS_Never);

  const auto DefaultLineEnding = FormatStyle::LE_DeriveLF;
  CHECK_PARSE("LineEnding: LF", LineEnding, FormatStyle::LE_LF);
  CHECK_PARSE("LineEnding: CRLF", LineEnding, FormatStyle::LE_CRLF);
  CHECK_PARSE("LineEnding: DeriveCRLF", LineEnding, FormatStyle::LE_DeriveCRLF);
  CHECK_PARSE("LineEnding: DeriveLF", LineEnding, DefaultLineEnding);
  // For backward compatibility:
  CHECK_PARSE("DeriveLineEnding: false", LineEnding, FormatStyle::LE_LF);
  Style.LineEnding = DefaultLineEnding;
  CHECK_PARSE("DeriveLineEnding: false\n"
              "UseCRLF: true",
              LineEnding, FormatStyle::LE_CRLF);
  Style.LineEnding = DefaultLineEnding;
  CHECK_PARSE("UseCRLF: true", LineEnding, FormatStyle::LE_DeriveCRLF);

  CHECK_PARSE("RemoveParentheses: MultipleParentheses", RemoveParentheses,
              FormatStyle::RPS_MultipleParentheses);
  CHECK_PARSE("RemoveParentheses: ReturnStatement", RemoveParentheses,
              FormatStyle::RPS_ReturnStatement);
  CHECK_PARSE("RemoveParentheses: Leave", RemoveParentheses,
              FormatStyle::RPS_Leave);

  CHECK_PARSE("AllowBreakBeforeNoexceptSpecifier: Always",
              AllowBreakBeforeNoexceptSpecifier, FormatStyle::BBNSS_Always);
  CHECK_PARSE("AllowBreakBeforeNoexceptSpecifier: OnlyWithParen",
              AllowBreakBeforeNoexceptSpecifier,
              FormatStyle::BBNSS_OnlyWithParen);
  CHECK_PARSE("AllowBreakBeforeNoexceptSpecifier: Never",
              AllowBreakBeforeNoexceptSpecifier, FormatStyle::BBNSS_Never);

  Style.SeparateDefinitionBlocks = FormatStyle::SDS_Never;
  CHECK_PARSE("SeparateDefinitionBlocks: Always", SeparateDefinitionBlocks,
              FormatStyle::SDS_Always);
  CHECK_PARSE("SeparateDefinitionBlocks: Leave", SeparateDefinitionBlocks,
              FormatStyle::SDS_Leave);
  CHECK_PARSE("SeparateDefinitionBlocks: Never", SeparateDefinitionBlocks,
              FormatStyle::SDS_Never);
}

TEST(ConfigParseTest, ParsesConfigurationWithLanguages) {
  FormatStyle Style = {};
  Style.Language = FormatStyle::LK_Cpp;
  CHECK_PARSE("Language: Cpp\n"
              "IndentWidth: 12",
              IndentWidth, 12u);
  EXPECT_EQ(parseConfiguration("Language: JavaScript\n"
                               "IndentWidth: 34",
                               &Style),
            ParseError::Unsuitable);
  FormatStyle BinPackedTCS = {};
  BinPackedTCS.Language = FormatStyle::LK_JavaScript;
  EXPECT_EQ(parseConfiguration("BinPackArguments: true\n"
                               "InsertTrailingCommas: Wrapped",
                               &BinPackedTCS),
            ParseError::BinPackTrailingCommaConflict);
  EXPECT_EQ(12u, Style.IndentWidth);
  CHECK_PARSE("IndentWidth: 56", IndentWidth, 56u);
  EXPECT_EQ(FormatStyle::LK_Cpp, Style.Language);

  Style.Language = FormatStyle::LK_JavaScript;
  CHECK_PARSE("Language: JavaScript\n"
              "IndentWidth: 12",
              IndentWidth, 12u);
  CHECK_PARSE("IndentWidth: 23", IndentWidth, 23u);
  EXPECT_EQ(parseConfiguration("Language: Cpp\n"
                               "IndentWidth: 34",
                               &Style),
            ParseError::Unsuitable);
  EXPECT_EQ(23u, Style.IndentWidth);
  CHECK_PARSE("IndentWidth: 56", IndentWidth, 56u);
  EXPECT_EQ(FormatStyle::LK_JavaScript, Style.Language);

  CHECK_PARSE("BasedOnStyle: LLVM\n"
              "IndentWidth: 67",
              IndentWidth, 67u);

  CHECK_PARSE("---\n"
              "Language: JavaScript\n"
              "IndentWidth: 12\n"
              "---\n"
              "Language: Cpp\n"
              "IndentWidth: 34\n"
              "...\n",
              IndentWidth, 12u);

  Style.Language = FormatStyle::LK_Cpp;
  CHECK_PARSE("---\n"
              "Language: JavaScript\n"
              "IndentWidth: 12\n"
              "---\n"
              "Language: Cpp\n"
              "IndentWidth: 34\n"
              "...\n",
              IndentWidth, 34u);
  CHECK_PARSE("---\n"
              "IndentWidth: 78\n"
              "---\n"
              "Language: JavaScript\n"
              "IndentWidth: 56\n"
              "...\n",
              IndentWidth, 78u);

  Style.ColumnLimit = 123;
  Style.IndentWidth = 234;
  Style.BreakBeforeBraces = FormatStyle::BS_Linux;
  Style.TabWidth = 345;
  EXPECT_FALSE(parseConfiguration("---\n"
                                  "IndentWidth: 456\n"
                                  "BreakBeforeBraces: Allman\n"
                                  "---\n"
                                  "Language: JavaScript\n"
                                  "IndentWidth: 111\n"
                                  "TabWidth: 111\n"
                                  "---\n"
                                  "Language: Cpp\n"
                                  "BreakBeforeBraces: Stroustrup\n"
                                  "TabWidth: 789\n"
                                  "...\n",
                                  &Style));
  EXPECT_EQ(123u, Style.ColumnLimit);
  EXPECT_EQ(456u, Style.IndentWidth);
  EXPECT_EQ(FormatStyle::BS_Stroustrup, Style.BreakBeforeBraces);
  EXPECT_EQ(789u, Style.TabWidth);

  EXPECT_EQ(parseConfiguration("---\n"
                               "Language: JavaScript\n"
                               "IndentWidth: 56\n"
                               "---\n"
                               "IndentWidth: 78\n"
                               "...\n",
                               &Style),
            ParseError::Error);
  EXPECT_EQ(parseConfiguration("---\n"
                               "Language: JavaScript\n"
                               "IndentWidth: 56\n"
                               "---\n"
                               "Language: JavaScript\n"
                               "IndentWidth: 78\n"
                               "...\n",
                               &Style),
            ParseError::Error);

  EXPECT_EQ(FormatStyle::LK_Cpp, Style.Language);

  Style.Language = FormatStyle::LK_Verilog;
  CHECK_PARSE("---\n"
              "Language: Verilog\n"
              "IndentWidth: 12\n"
              "---\n"
              "Language: Cpp\n"
              "IndentWidth: 34\n"
              "...\n",
              IndentWidth, 12u);
  CHECK_PARSE("---\n"
              "IndentWidth: 78\n"
              "---\n"
              "Language: Verilog\n"
              "IndentWidth: 56\n"
              "...\n",
              IndentWidth, 56u);
}

TEST(ConfigParseTest, UsesLanguageForBasedOnStyle) {
  FormatStyle Style = {};
  Style.Language = FormatStyle::LK_JavaScript;
  Style.BreakBeforeTernaryOperators = true;
  EXPECT_EQ(0, parseConfiguration("BasedOnStyle: Google", &Style).value());
  EXPECT_FALSE(Style.BreakBeforeTernaryOperators);

  Style.BreakBeforeTernaryOperators = true;
  EXPECT_EQ(0, parseConfiguration("---\n"
                                  "BasedOnStyle: Google\n"
                                  "---\n"
                                  "Language: JavaScript\n"
                                  "IndentWidth: 76\n"
                                  "...\n",
                                  &Style)
                   .value());
  EXPECT_FALSE(Style.BreakBeforeTernaryOperators);
  EXPECT_EQ(76u, Style.IndentWidth);
  EXPECT_EQ(FormatStyle::LK_JavaScript, Style.Language);
}

TEST(ConfigParseTest, ConfigurationRoundTripTest) {
  FormatStyle Style = getLLVMStyle();
  std::string YAML = configurationAsText(Style);
  FormatStyle ParsedStyle = {};
  ParsedStyle.Language = FormatStyle::LK_Cpp;
  EXPECT_EQ(0, parseConfiguration(YAML, &ParsedStyle).value());
  EXPECT_EQ(Style, ParsedStyle);
}

TEST(ConfigParseTest, GetStyleWithEmptyFileName) {
  llvm::vfs::InMemoryFileSystem FS;
  auto Style1 = getStyle("file", "", "Google", "", &FS);
  ASSERT_TRUE((bool)Style1);
  ASSERT_EQ(*Style1, getGoogleStyle());
}

TEST(ConfigParseTest, GetStyleOfFile) {
  llvm::vfs::InMemoryFileSystem FS;
  // Test 1: format file in the same directory.
  ASSERT_TRUE(
      FS.addFile("/a/.clang-format", 0,
                 llvm::MemoryBuffer::getMemBuffer("BasedOnStyle: LLVM")));
  ASSERT_TRUE(
      FS.addFile("/a/test.cpp", 0, llvm::MemoryBuffer::getMemBuffer("int i;")));
  auto Style1 = getStyle("file", "/a/.clang-format", "Google", "", &FS);
  ASSERT_TRUE((bool)Style1);
  ASSERT_EQ(*Style1, getLLVMStyle());

  // Test 2.1: fallback to default.
  ASSERT_TRUE(
      FS.addFile("/b/test.cpp", 0, llvm::MemoryBuffer::getMemBuffer("int i;")));
  auto Style2 = getStyle("file", "/b/test.cpp", "Mozilla", "", &FS);
  ASSERT_TRUE((bool)Style2);
  ASSERT_EQ(*Style2, getMozillaStyle());

  // Test 2.2: no format on 'none' fallback style.
  Style2 = getStyle("file", "/b/test.cpp", "none", "", &FS);
  ASSERT_TRUE((bool)Style2);
  ASSERT_EQ(*Style2, getNoStyle());

  // Test 2.3: format if config is found with no based style while fallback is
  // 'none'.
  ASSERT_TRUE(FS.addFile("/b/.clang-format", 0,
                         llvm::MemoryBuffer::getMemBuffer("IndentWidth: 2")));
  Style2 = getStyle("file", "/b/test.cpp", "none", "", &FS);
  ASSERT_TRUE((bool)Style2);
  ASSERT_EQ(*Style2, getLLVMStyle());

  // Test 2.4: format if yaml with no based style, while fallback is 'none'.
  Style2 = getStyle("{}", "a.h", "none", "", &FS);
  ASSERT_TRUE((bool)Style2);
  ASSERT_EQ(*Style2, getLLVMStyle());

  // Test 3: format file in parent directory.
  ASSERT_TRUE(
      FS.addFile("/c/.clang-format", 0,
                 llvm::MemoryBuffer::getMemBuffer("BasedOnStyle: Google")));
  ASSERT_TRUE(FS.addFile("/c/sub/sub/sub/test.cpp", 0,
                         llvm::MemoryBuffer::getMemBuffer("int i;")));
  auto Style3 = getStyle("file", "/c/sub/sub/sub/test.cpp", "LLVM", "", &FS);
  ASSERT_TRUE((bool)Style3);
  ASSERT_EQ(*Style3, getGoogleStyle());

  // Test 4: error on invalid fallback style
  auto Style4 = getStyle("file", "a.h", "KungFu", "", &FS);
  ASSERT_FALSE((bool)Style4);
  llvm::consumeError(Style4.takeError());

  // Test 5: error on invalid yaml on command line
  auto Style5 = getStyle("{invalid_key=invalid_value}", "a.h", "LLVM", "", &FS);
  ASSERT_FALSE((bool)Style5);
  llvm::consumeError(Style5.takeError());

  // Test 6: error on invalid style
  auto Style6 = getStyle("KungFu", "a.h", "LLVM", "", &FS);
  ASSERT_FALSE((bool)Style6);
  llvm::consumeError(Style6.takeError());

  // Test 7: found config file, error on parsing it
  ASSERT_TRUE(
      FS.addFile("/d/.clang-format", 0,
                 llvm::MemoryBuffer::getMemBuffer("BasedOnStyle: LLVM\n"
                                                  "InvalidKey: InvalidValue")));
  ASSERT_TRUE(
      FS.addFile("/d/test.cpp", 0, llvm::MemoryBuffer::getMemBuffer("int i;")));
  auto Style7a = getStyle("file", "/d/.clang-format", "LLVM", "", &FS);
  ASSERT_FALSE((bool)Style7a);
  llvm::consumeError(Style7a.takeError());

  auto Style7b = getStyle("file", "/d/.clang-format", "LLVM", "", &FS, true);
  ASSERT_TRUE((bool)Style7b);

  // Test 8: inferred per-language defaults apply.
  auto StyleTd = getStyle("file", "x.td", "llvm", "", &FS);
  ASSERT_TRUE((bool)StyleTd);
  ASSERT_EQ(*StyleTd, getLLVMStyle(FormatStyle::LK_TableGen));

  // Test 9.1.1: overwriting a file style, when no parent file exists with no
  // fallback style.
  ASSERT_TRUE(FS.addFile(
      "/e/sub/.clang-format", 0,
      llvm::MemoryBuffer::getMemBuffer("BasedOnStyle: InheritParentConfig\n"
                                       "ColumnLimit: 20")));
  ASSERT_TRUE(FS.addFile("/e/sub/code.cpp", 0,
                         llvm::MemoryBuffer::getMemBuffer("int i;")));
  auto Style9 = getStyle("file", "/e/sub/code.cpp", "none", "", &FS);
  ASSERT_TRUE(static_cast<bool>(Style9));
  ASSERT_EQ(*Style9, [] {
    auto Style = getNoStyle();
    Style.ColumnLimit = 20;
    return Style;
  }());

  // Test 9.1.2: propagate more than one level with no parent file.
  ASSERT_TRUE(FS.addFile("/e/sub/sub/code.cpp", 0,
                         llvm::MemoryBuffer::getMemBuffer("int i;")));
  ASSERT_TRUE(FS.addFile("/e/sub/sub/.clang-format", 0,
                         llvm::MemoryBuffer::getMemBuffer(
                             "BasedOnStyle: InheritParentConfig\n"
                             "WhitespaceSensitiveMacros: ['FOO', 'BAR']")));
  std::vector<std::string> NonDefaultWhiteSpaceMacros =
      Style9->WhitespaceSensitiveMacros;
  NonDefaultWhiteSpaceMacros[0] = "FOO";
  NonDefaultWhiteSpaceMacros[1] = "BAR";

  ASSERT_NE(Style9->WhitespaceSensitiveMacros, NonDefaultWhiteSpaceMacros);
  Style9 = getStyle("file", "/e/sub/sub/code.cpp", "none", "", &FS);
  ASSERT_TRUE(static_cast<bool>(Style9));
  ASSERT_EQ(*Style9, [&NonDefaultWhiteSpaceMacros] {
    auto Style = getNoStyle();
    Style.ColumnLimit = 20;
    Style.WhitespaceSensitiveMacros = NonDefaultWhiteSpaceMacros;
    return Style;
  }());

  // Test 9.2: with LLVM fallback style
  Style9 = getStyle("file", "/e/sub/code.cpp", "LLVM", "", &FS);
  ASSERT_TRUE(static_cast<bool>(Style9));
  ASSERT_EQ(*Style9, [] {
    auto Style = getLLVMStyle();
    Style.ColumnLimit = 20;
    return Style;
  }());

  // Test 9.3: with a parent file
  ASSERT_TRUE(
      FS.addFile("/e/.clang-format", 0,
                 llvm::MemoryBuffer::getMemBuffer("BasedOnStyle: Google\n"
                                                  "UseTab: Always")));
  Style9 = getStyle("file", "/e/sub/code.cpp", "none", "", &FS);
  ASSERT_TRUE(static_cast<bool>(Style9));
  ASSERT_EQ(*Style9, [] {
    auto Style = getGoogleStyle();
    Style.ColumnLimit = 20;
    Style.UseTab = FormatStyle::UT_Always;
    return Style;
  }());

  // Test 9.4: propagate more than one level with a parent file.
  const auto SubSubStyle = [&NonDefaultWhiteSpaceMacros] {
    auto Style = getGoogleStyle();
    Style.ColumnLimit = 20;
    Style.UseTab = FormatStyle::UT_Always;
    Style.WhitespaceSensitiveMacros = NonDefaultWhiteSpaceMacros;
    return Style;
  }();

  ASSERT_NE(Style9->WhitespaceSensitiveMacros, NonDefaultWhiteSpaceMacros);
  Style9 = getStyle("file", "/e/sub/sub/code.cpp", "none", "", &FS);
  ASSERT_TRUE(static_cast<bool>(Style9));
  ASSERT_EQ(*Style9, SubSubStyle);

  // Test 9.5: use InheritParentConfig as style name
  Style9 =
      getStyle("inheritparentconfig", "/e/sub/sub/code.cpp", "none", "", &FS);
  ASSERT_TRUE(static_cast<bool>(Style9));
  ASSERT_EQ(*Style9, SubSubStyle);

  // Test 9.6: use command line style with inheritance
  Style9 = getStyle("{BasedOnStyle: InheritParentConfig}",
                    "/e/sub/sub/code.cpp", "none", "", &FS);
  ASSERT_TRUE(static_cast<bool>(Style9));
  ASSERT_EQ(*Style9, SubSubStyle);

  // Test 9.7: use command line style with inheritance and own config
  Style9 = getStyle("{BasedOnStyle: InheritParentConfig, "
                    "WhitespaceSensitiveMacros: ['FOO', 'BAR']}",
                    "/e/sub/code.cpp", "none", "", &FS);
  ASSERT_TRUE(static_cast<bool>(Style9));
  ASSERT_EQ(*Style9, SubSubStyle);

  // Test 9.8: use inheritance from a file without BasedOnStyle
  ASSERT_TRUE(FS.addFile("/e/withoutbase/.clang-format", 0,
                         llvm::MemoryBuffer::getMemBuffer("ColumnLimit: 123")));
  ASSERT_TRUE(
      FS.addFile("/e/withoutbase/sub/.clang-format", 0,
                 llvm::MemoryBuffer::getMemBuffer(
                     "BasedOnStyle: InheritParentConfig\nIndentWidth: 7")));
  // Make sure we do not use the fallback style
  Style9 = getStyle("file", "/e/withoutbase/code.cpp", "google", "", &FS);
  ASSERT_TRUE(static_cast<bool>(Style9));
  ASSERT_EQ(*Style9, [] {
    auto Style = getLLVMStyle();
    Style.ColumnLimit = 123;
    return Style;
  }());

  Style9 = getStyle("file", "/e/withoutbase/sub/code.cpp", "google", "", &FS);
  ASSERT_TRUE(static_cast<bool>(Style9));
  ASSERT_EQ(*Style9, [] {
    auto Style = getLLVMStyle();
    Style.ColumnLimit = 123;
    Style.IndentWidth = 7;
    return Style;
  }());

  // Test 9.9: use inheritance from a specific config file.
  Style9 = getStyle("file:/e/sub/sub/.clang-format", "/e/sub/sub/code.cpp",
                    "none", "", &FS);
  ASSERT_TRUE(static_cast<bool>(Style9));
  ASSERT_EQ(*Style9, SubSubStyle);
}

TEST(ConfigParseTest, GetStyleOfSpecificFile) {
  llvm::vfs::InMemoryFileSystem FS;
  // Specify absolute path to a format file in a parent directory.
  ASSERT_TRUE(
      FS.addFile("/e/.clang-format", 0,
                 llvm::MemoryBuffer::getMemBuffer("BasedOnStyle: LLVM")));
  ASSERT_TRUE(
      FS.addFile("/e/explicit.clang-format", 0,
                 llvm::MemoryBuffer::getMemBuffer("BasedOnStyle: Google")));
  ASSERT_TRUE(FS.addFile("/e/sub/sub/sub/test.cpp", 0,
                         llvm::MemoryBuffer::getMemBuffer("int i;")));
  auto Style = getStyle("file:/e/explicit.clang-format",
                        "/e/sub/sub/sub/test.cpp", "LLVM", "", &FS);
  ASSERT_TRUE(static_cast<bool>(Style));
  ASSERT_EQ(*Style, getGoogleStyle());

  // Specify relative path to a format file.
  ASSERT_TRUE(
      FS.addFile("../../e/explicit.clang-format", 0,
                 llvm::MemoryBuffer::getMemBuffer("BasedOnStyle: Google")));
  Style = getStyle("file:../../e/explicit.clang-format",
                   "/e/sub/sub/sub/test.cpp", "LLVM", "", &FS);
  ASSERT_TRUE(static_cast<bool>(Style));
  ASSERT_EQ(*Style, getGoogleStyle());

  // Specify path to a format file that does not exist.
  Style = getStyle("file:/e/missing.clang-format", "/e/sub/sub/sub/test.cpp",
                   "LLVM", "", &FS);
  ASSERT_FALSE(static_cast<bool>(Style));
  llvm::consumeError(Style.takeError());

  // Specify path to a file on the filesystem.
  SmallString<128> FormatFilePath;
  std::error_code ECF = llvm::sys::fs::createTemporaryFile(
      "FormatFileTest", "tpl", FormatFilePath);
  EXPECT_FALSE((bool)ECF);
  llvm::raw_fd_ostream FormatFileTest(FormatFilePath, ECF);
  EXPECT_FALSE((bool)ECF);
  FormatFileTest << "BasedOnStyle: Google\n";
  FormatFileTest.close();

  SmallString<128> TestFilePath;
  std::error_code ECT =
      llvm::sys::fs::createTemporaryFile("CodeFileTest", "cc", TestFilePath);
  EXPECT_FALSE((bool)ECT);
  llvm::raw_fd_ostream CodeFileTest(TestFilePath, ECT);
  CodeFileTest << "int i;\n";
  CodeFileTest.close();

  std::string format_file_arg = std::string("file:") + FormatFilePath.c_str();
  Style = getStyle(format_file_arg, TestFilePath, "LLVM", "", nullptr);

  llvm::sys::fs::remove(FormatFilePath.c_str());
  llvm::sys::fs::remove(TestFilePath.c_str());
  ASSERT_TRUE(static_cast<bool>(Style));
  ASSERT_EQ(*Style, getGoogleStyle());
}

} // namespace
} // namespace format
} // namespace clang
