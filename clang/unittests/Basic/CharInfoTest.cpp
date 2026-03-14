//===- unittests/Basic/CharInfoTest.cpp -- ASCII classification tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/CharInfo.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

// Check that the CharInfo table has been constructed reasonably.
TEST(CharInfoTest, validateInfoTable) {
  using namespace charinfo;
  EXPECT_EQ((unsigned)CHAR_SPACE,   InfoTable[(unsigned)' ']);
  EXPECT_EQ((unsigned)CHAR_HORZ_WS, InfoTable[(unsigned)'\t']);
  EXPECT_EQ((unsigned)CHAR_HORZ_WS, InfoTable[(unsigned)'\f']); // ??
  EXPECT_EQ((unsigned)CHAR_HORZ_WS, InfoTable[(unsigned)'\v']); // ??
  EXPECT_EQ((unsigned)CHAR_VERT_WS, InfoTable[(unsigned)'\n']);
  EXPECT_EQ((unsigned)CHAR_VERT_WS, InfoTable[(unsigned)'\r']);
  EXPECT_EQ((unsigned)CHAR_UNDER,   InfoTable[(unsigned)'_']);
  EXPECT_EQ((unsigned)CHAR_PERIOD,  InfoTable[(unsigned)'.']);

  for (unsigned i = 'a'; i <= 'f'; ++i) {
    EXPECT_EQ((unsigned)CHAR_XLOWER, InfoTable[i]);
    EXPECT_EQ((unsigned)CHAR_XUPPER, InfoTable[i+'A'-'a']);
  }

  for (unsigned i = 'g'; i <= 'z'; ++i) {
    EXPECT_EQ((unsigned)CHAR_LOWER, InfoTable[i]);
    EXPECT_EQ((unsigned)CHAR_UPPER, InfoTable[i+'A'-'a']);
  }

  for (unsigned i = '0'; i <= '9'; ++i)
    EXPECT_EQ((unsigned)CHAR_DIGIT, InfoTable[i]);
}

// Check various predicates.
TEST(CharInfoTest, isASCII) {
  EXPECT_TRUE(isASCII('\0'));
  EXPECT_TRUE(isASCII('\n'));
  EXPECT_TRUE(isASCII(' '));
  EXPECT_TRUE(isASCII('a'));
  EXPECT_TRUE(isASCII('\x7f'));
  EXPECT_FALSE(isASCII('\x80'));
  EXPECT_FALSE(isASCII('\xc2'));
  EXPECT_FALSE(isASCII('\xff'));
}

TEST(CharInfoTest, isAsciiIdentifierStart) {
  EXPECT_TRUE(isAsciiIdentifierStart('a'));
  EXPECT_TRUE(isAsciiIdentifierStart('A'));
  EXPECT_TRUE(isAsciiIdentifierStart('z'));
  EXPECT_TRUE(isAsciiIdentifierStart('Z'));
  EXPECT_TRUE(isAsciiIdentifierStart('_'));

  EXPECT_FALSE(isAsciiIdentifierStart('0'));
  EXPECT_FALSE(isAsciiIdentifierStart('.'));
  EXPECT_FALSE(isAsciiIdentifierStart('`'));
  EXPECT_FALSE(isAsciiIdentifierStart('\0'));

  EXPECT_FALSE(isAsciiIdentifierStart('$'));
  EXPECT_TRUE(isAsciiIdentifierStart('$', /*AllowDollar=*/true));

  EXPECT_FALSE(isAsciiIdentifierStart('\x80'));
  EXPECT_FALSE(isAsciiIdentifierStart('\xc2'));
  EXPECT_FALSE(isAsciiIdentifierStart('\xff'));
}

TEST(CharInfoTest, isAsciiIdentifierContinue) {
  EXPECT_TRUE(isAsciiIdentifierContinue('a'));
  EXPECT_TRUE(isAsciiIdentifierContinue('A'));
  EXPECT_TRUE(isAsciiIdentifierContinue('z'));
  EXPECT_TRUE(isAsciiIdentifierContinue('Z'));
  EXPECT_TRUE(isAsciiIdentifierContinue('_'));

  EXPECT_TRUE(isAsciiIdentifierContinue('0'));
  EXPECT_FALSE(isAsciiIdentifierContinue('.'));
  EXPECT_FALSE(isAsciiIdentifierContinue('`'));
  EXPECT_FALSE(isAsciiIdentifierContinue('\0'));

  EXPECT_FALSE(isAsciiIdentifierContinue('$'));
  EXPECT_TRUE(isAsciiIdentifierContinue('$', /*AllowDollar=*/true));

  EXPECT_FALSE(isAsciiIdentifierContinue('\x80'));
  EXPECT_FALSE(isAsciiIdentifierContinue('\xc2'));
  EXPECT_FALSE(isAsciiIdentifierContinue('\xff'));
}

TEST(CharInfoTest, isHorizontalWhitespace) {
  EXPECT_FALSE(isHorizontalWhitespace('a'));
  EXPECT_FALSE(isHorizontalWhitespace('_'));
  EXPECT_FALSE(isHorizontalWhitespace('0'));
  EXPECT_FALSE(isHorizontalWhitespace('.'));
  EXPECT_FALSE(isHorizontalWhitespace('`'));
  EXPECT_FALSE(isHorizontalWhitespace('\0'));
  EXPECT_FALSE(isHorizontalWhitespace('\x7f'));

  EXPECT_TRUE(isHorizontalWhitespace(' '));
  EXPECT_TRUE(isHorizontalWhitespace('\t'));
  EXPECT_TRUE(isHorizontalWhitespace('\f')); // ??
  EXPECT_TRUE(isHorizontalWhitespace('\v')); // ??

  EXPECT_FALSE(isHorizontalWhitespace('\n'));
  EXPECT_FALSE(isHorizontalWhitespace('\r'));

  EXPECT_FALSE(isHorizontalWhitespace('\x80'));
  EXPECT_FALSE(isHorizontalWhitespace('\xc2'));
  EXPECT_FALSE(isHorizontalWhitespace('\xff'));
}

TEST(CharInfoTest, isVerticalWhitespace) {
  EXPECT_FALSE(isVerticalWhitespace('a'));
  EXPECT_FALSE(isVerticalWhitespace('_'));
  EXPECT_FALSE(isVerticalWhitespace('0'));
  EXPECT_FALSE(isVerticalWhitespace('.'));
  EXPECT_FALSE(isVerticalWhitespace('`'));
  EXPECT_FALSE(isVerticalWhitespace('\0'));
  EXPECT_FALSE(isVerticalWhitespace('\x7f'));

  EXPECT_FALSE(isVerticalWhitespace(' '));
  EXPECT_FALSE(isVerticalWhitespace('\t'));
  EXPECT_FALSE(isVerticalWhitespace('\f')); // ??
  EXPECT_FALSE(isVerticalWhitespace('\v')); // ??

  EXPECT_TRUE(isVerticalWhitespace('\n'));
  EXPECT_TRUE(isVerticalWhitespace('\r'));

  EXPECT_FALSE(isVerticalWhitespace('\x80'));
  EXPECT_FALSE(isVerticalWhitespace('\xc2'));
  EXPECT_FALSE(isVerticalWhitespace('\xff'));
}

TEST(CharInfoTest, isWhitespace) {
  EXPECT_FALSE(isWhitespace('a'));
  EXPECT_FALSE(isWhitespace('_'));
  EXPECT_FALSE(isWhitespace('0'));
  EXPECT_FALSE(isWhitespace('.'));
  EXPECT_FALSE(isWhitespace('`'));
  EXPECT_FALSE(isWhitespace('\0'));
  EXPECT_FALSE(isWhitespace('\x7f'));

  EXPECT_TRUE(isWhitespace(' '));
  EXPECT_TRUE(isWhitespace('\t'));
  EXPECT_TRUE(isWhitespace('\f'));
  EXPECT_TRUE(isWhitespace('\v'));

  EXPECT_TRUE(isWhitespace('\n'));
  EXPECT_TRUE(isWhitespace('\r'));

  EXPECT_FALSE(isWhitespace('\x80'));
  EXPECT_FALSE(isWhitespace('\xc2'));
  EXPECT_FALSE(isWhitespace('\xff'));
}

TEST(CharInfoTest, isDigit) {
  EXPECT_TRUE(isDigit('0'));
  EXPECT_TRUE(isDigit('9'));

  EXPECT_FALSE(isDigit('a'));
  EXPECT_FALSE(isDigit('A'));

  EXPECT_FALSE(isDigit('z'));
  EXPECT_FALSE(isDigit('Z'));
  
  EXPECT_FALSE(isDigit('.'));
  EXPECT_FALSE(isDigit('_'));

  EXPECT_FALSE(isDigit('/'));
  EXPECT_FALSE(isDigit('\0'));

  EXPECT_FALSE(isDigit('\x80'));
  EXPECT_FALSE(isDigit('\xc2'));
  EXPECT_FALSE(isDigit('\xff'));
}

TEST(CharInfoTest, isHexDigit) {
  EXPECT_TRUE(isHexDigit('0'));
  EXPECT_TRUE(isHexDigit('9'));

  EXPECT_TRUE(isHexDigit('a'));
  EXPECT_TRUE(isHexDigit('A'));

  EXPECT_FALSE(isHexDigit('z'));
  EXPECT_FALSE(isHexDigit('Z'));
  
  EXPECT_FALSE(isHexDigit('.'));
  EXPECT_FALSE(isHexDigit('_'));

  EXPECT_FALSE(isHexDigit('/'));
  EXPECT_FALSE(isHexDigit('\0'));

  EXPECT_FALSE(isHexDigit('\x80'));
  EXPECT_FALSE(isHexDigit('\xc2'));
  EXPECT_FALSE(isHexDigit('\xff'));
}

TEST(CharInfoTest, isLetter) {
  EXPECT_FALSE(isLetter('0'));
  EXPECT_FALSE(isLetter('9'));

  EXPECT_TRUE(isLetter('a'));
  EXPECT_TRUE(isLetter('A'));

  EXPECT_TRUE(isLetter('z'));
  EXPECT_TRUE(isLetter('Z'));
  
  EXPECT_FALSE(isLetter('.'));
  EXPECT_FALSE(isLetter('_'));

  EXPECT_FALSE(isLetter('/'));
  EXPECT_FALSE(isLetter('('));
  EXPECT_FALSE(isLetter('\0'));

  EXPECT_FALSE(isLetter('\x80'));
  EXPECT_FALSE(isLetter('\xc2'));
  EXPECT_FALSE(isLetter('\xff'));
}

TEST(CharInfoTest, isLowercase) {
  EXPECT_FALSE(isLowercase('0'));
  EXPECT_FALSE(isLowercase('9'));

  EXPECT_TRUE(isLowercase('a'));
  EXPECT_FALSE(isLowercase('A'));

  EXPECT_TRUE(isLowercase('z'));
  EXPECT_FALSE(isLowercase('Z'));
  
  EXPECT_FALSE(isLowercase('.'));
  EXPECT_FALSE(isLowercase('_'));

  EXPECT_FALSE(isLowercase('/'));
  EXPECT_FALSE(isLowercase('('));
  EXPECT_FALSE(isLowercase('\0'));

  EXPECT_FALSE(isLowercase('\x80'));
  EXPECT_FALSE(isLowercase('\xc2'));
  EXPECT_FALSE(isLowercase('\xff'));
}

TEST(CharInfoTest, isUppercase) {
  EXPECT_FALSE(isUppercase('0'));
  EXPECT_FALSE(isUppercase('9'));

  EXPECT_FALSE(isUppercase('a'));
  EXPECT_TRUE(isUppercase('A'));

  EXPECT_FALSE(isUppercase('z'));
  EXPECT_TRUE(isUppercase('Z'));

  EXPECT_FALSE(isUppercase('.'));
  EXPECT_FALSE(isUppercase('_'));

  EXPECT_FALSE(isUppercase('/'));
  EXPECT_FALSE(isUppercase('('));
  EXPECT_FALSE(isUppercase('\0'));

  EXPECT_FALSE(isUppercase('\x80'));
  EXPECT_FALSE(isUppercase('\xc2'));
  EXPECT_FALSE(isUppercase('\xff'));
}

TEST(CharInfoTest, isAlphanumeric) {
  EXPECT_TRUE(isAlphanumeric('0'));
  EXPECT_TRUE(isAlphanumeric('9'));

  EXPECT_TRUE(isAlphanumeric('a'));
  EXPECT_TRUE(isAlphanumeric('A'));

  EXPECT_TRUE(isAlphanumeric('z'));
  EXPECT_TRUE(isAlphanumeric('Z'));

  EXPECT_FALSE(isAlphanumeric('.'));
  EXPECT_FALSE(isAlphanumeric('_'));

  EXPECT_FALSE(isAlphanumeric('/'));
  EXPECT_FALSE(isAlphanumeric('('));
  EXPECT_FALSE(isAlphanumeric('\0'));

  EXPECT_FALSE(isAlphanumeric('\x80'));
  EXPECT_FALSE(isAlphanumeric('\xc2'));
  EXPECT_FALSE(isAlphanumeric('\xff'));
}

TEST(CharInfoTest, isPunctuation) {
  EXPECT_FALSE(isPunctuation('0'));
  EXPECT_FALSE(isPunctuation('9'));

  EXPECT_FALSE(isPunctuation('a'));
  EXPECT_FALSE(isPunctuation('A'));

  EXPECT_FALSE(isPunctuation('z'));
  EXPECT_FALSE(isPunctuation('Z'));

  EXPECT_TRUE(isPunctuation('.'));
  EXPECT_TRUE(isPunctuation('_'));

  EXPECT_TRUE(isPunctuation('/'));
  EXPECT_TRUE(isPunctuation('('));

  EXPECT_FALSE(isPunctuation(' '));
  EXPECT_FALSE(isPunctuation('\n'));
  EXPECT_FALSE(isPunctuation('\0'));

  EXPECT_FALSE(isPunctuation('\x80'));
  EXPECT_FALSE(isPunctuation('\xc2'));
  EXPECT_FALSE(isPunctuation('\xff'));
}

TEST(CharInfoTest, isPrintable) {
  EXPECT_TRUE(isPrintable('0'));
  EXPECT_TRUE(isPrintable('9'));

  EXPECT_TRUE(isPrintable('a'));
  EXPECT_TRUE(isPrintable('A'));

  EXPECT_TRUE(isPrintable('z'));
  EXPECT_TRUE(isPrintable('Z'));

  EXPECT_TRUE(isPrintable('.'));
  EXPECT_TRUE(isPrintable('_'));

  EXPECT_TRUE(isPrintable('/'));
  EXPECT_TRUE(isPrintable('('));

  EXPECT_TRUE(isPrintable(' '));
  EXPECT_FALSE(isPrintable('\t'));
  EXPECT_FALSE(isPrintable('\n'));
  EXPECT_FALSE(isPrintable('\0'));

  EXPECT_FALSE(isPrintable('\x80'));
  EXPECT_FALSE(isPrintable('\xc2'));
  EXPECT_FALSE(isPrintable('\xff'));
}

TEST(CharInfoTest, isPreprocessingNumberBody) {
  EXPECT_TRUE(isPreprocessingNumberBody('0'));
  EXPECT_TRUE(isPreprocessingNumberBody('9'));

  EXPECT_TRUE(isPreprocessingNumberBody('a'));
  EXPECT_TRUE(isPreprocessingNumberBody('A'));

  EXPECT_TRUE(isPreprocessingNumberBody('z'));
  EXPECT_TRUE(isPreprocessingNumberBody('Z'));
  EXPECT_TRUE(isPreprocessingNumberBody('.'));
  EXPECT_TRUE(isPreprocessingNumberBody('_'));

  EXPECT_FALSE(isPreprocessingNumberBody('/'));
  EXPECT_FALSE(isPreprocessingNumberBody('('));
  EXPECT_FALSE(isPreprocessingNumberBody('\0'));

  EXPECT_FALSE(isPreprocessingNumberBody('\x80'));
  EXPECT_FALSE(isPreprocessingNumberBody('\xc2'));
  EXPECT_FALSE(isPreprocessingNumberBody('\xff'));
}

TEST(CharInfoTest, isRawStringDelimBody) {
  EXPECT_TRUE(isRawStringDelimBody('0'));
  EXPECT_TRUE(isRawStringDelimBody('9'));

  EXPECT_TRUE(isRawStringDelimBody('a'));
  EXPECT_TRUE(isRawStringDelimBody('A'));

  EXPECT_TRUE(isRawStringDelimBody('z'));
  EXPECT_TRUE(isRawStringDelimBody('Z'));
  EXPECT_TRUE(isRawStringDelimBody('.'));
  EXPECT_TRUE(isRawStringDelimBody('_'));

  EXPECT_TRUE(isRawStringDelimBody('/'));
  EXPECT_FALSE(isRawStringDelimBody('('));
  EXPECT_FALSE(isRawStringDelimBody('\0'));

  EXPECT_FALSE(isRawStringDelimBody('\x80'));
  EXPECT_FALSE(isRawStringDelimBody('\xc2'));
  EXPECT_FALSE(isRawStringDelimBody('\xff'));
}

TEST(CharInfoTest, toLowercase) {
  EXPECT_EQ('0', toLowercase('0'));
  EXPECT_EQ('9', toLowercase('9'));

  EXPECT_EQ('a', toLowercase('a'));
  EXPECT_EQ('a', toLowercase('A'));

  EXPECT_EQ('z', toLowercase('z'));
  EXPECT_EQ('z', toLowercase('Z'));

  EXPECT_EQ('.', toLowercase('.'));
  EXPECT_EQ('_', toLowercase('_'));

  EXPECT_EQ('/', toLowercase('/'));
  EXPECT_EQ('\0', toLowercase('\0'));
}

TEST(CharInfoTest, toUppercase) {
  EXPECT_EQ('0', toUppercase('0'));
  EXPECT_EQ('9', toUppercase('9'));

  EXPECT_EQ('A', toUppercase('a'));
  EXPECT_EQ('A', toUppercase('A'));

  EXPECT_EQ('Z', toUppercase('z'));
  EXPECT_EQ('Z', toUppercase('Z'));

  EXPECT_EQ('.', toUppercase('.'));
  EXPECT_EQ('_', toUppercase('_'));

  EXPECT_EQ('/', toUppercase('/'));
  EXPECT_EQ('\0', toUppercase('\0'));
}

TEST(CharInfoTest, isValidAsciiIdentifier) {
  EXPECT_FALSE(isValidAsciiIdentifier(""));

  // 1 character
  EXPECT_FALSE(isValidAsciiIdentifier("."));
  EXPECT_FALSE(isValidAsciiIdentifier("\n"));
  EXPECT_FALSE(isValidAsciiIdentifier(" "));
  EXPECT_FALSE(isValidAsciiIdentifier("\x80"));
  EXPECT_FALSE(isValidAsciiIdentifier("\xc2"));
  EXPECT_FALSE(isValidAsciiIdentifier("\xff"));
  EXPECT_FALSE(isValidAsciiIdentifier("$"));
  EXPECT_FALSE(isValidAsciiIdentifier("1"));

  EXPECT_TRUE(isValidAsciiIdentifier("_"));
  EXPECT_TRUE(isValidAsciiIdentifier("a"));
  EXPECT_TRUE(isValidAsciiIdentifier("z"));
  EXPECT_TRUE(isValidAsciiIdentifier("A"));
  EXPECT_TRUE(isValidAsciiIdentifier("Z"));
  EXPECT_TRUE(isValidAsciiIdentifier("$", /*AllowDollar=*/true));

  // 2 characters, '_' suffix
  EXPECT_FALSE(isValidAsciiIdentifier("._"));
  EXPECT_FALSE(isValidAsciiIdentifier("\n_"));
  EXPECT_FALSE(isValidAsciiIdentifier(" _"));
  EXPECT_FALSE(isValidAsciiIdentifier("\x80_"));
  EXPECT_FALSE(isValidAsciiIdentifier("\xc2_"));
  EXPECT_FALSE(isValidAsciiIdentifier("\xff_"));
  EXPECT_FALSE(isValidAsciiIdentifier("$_"));
  EXPECT_FALSE(isValidAsciiIdentifier("1_"));

  EXPECT_TRUE(isValidAsciiIdentifier("__"));
  EXPECT_TRUE(isValidAsciiIdentifier("a_"));
  EXPECT_TRUE(isValidAsciiIdentifier("z_"));
  EXPECT_TRUE(isValidAsciiIdentifier("A_"));
  EXPECT_TRUE(isValidAsciiIdentifier("Z_"));
  EXPECT_TRUE(isValidAsciiIdentifier("$_", /*AllowDollar=*/true));

  // 2 characters, '_' prefix
  EXPECT_FALSE(isValidAsciiIdentifier("_."));
  EXPECT_FALSE(isValidAsciiIdentifier("_\n"));
  EXPECT_FALSE(isValidAsciiIdentifier("_ "));
  EXPECT_FALSE(isValidAsciiIdentifier("_\x80"));
  EXPECT_FALSE(isValidAsciiIdentifier("_\xc2"));
  EXPECT_FALSE(isValidAsciiIdentifier("_\xff"));
  EXPECT_FALSE(isValidAsciiIdentifier("_$"));
  EXPECT_TRUE(isValidAsciiIdentifier("_1"));

  EXPECT_TRUE(isValidAsciiIdentifier("__"));
  EXPECT_TRUE(isValidAsciiIdentifier("_a"));
  EXPECT_TRUE(isValidAsciiIdentifier("_z"));
  EXPECT_TRUE(isValidAsciiIdentifier("_A"));
  EXPECT_TRUE(isValidAsciiIdentifier("_Z"));
  EXPECT_TRUE(isValidAsciiIdentifier("_$", /*AllowDollar=*/true));

  // 3 characters, '__' prefix
  EXPECT_FALSE(isValidAsciiIdentifier("__."));
  EXPECT_FALSE(isValidAsciiIdentifier("__\n"));
  EXPECT_FALSE(isValidAsciiIdentifier("__ "));
  EXPECT_FALSE(isValidAsciiIdentifier("__\x80"));
  EXPECT_FALSE(isValidAsciiIdentifier("__\xc2"));
  EXPECT_FALSE(isValidAsciiIdentifier("__\xff"));
  EXPECT_FALSE(isValidAsciiIdentifier("__$"));
  EXPECT_TRUE(isValidAsciiIdentifier("__1"));

  EXPECT_TRUE(isValidAsciiIdentifier("___"));
  EXPECT_TRUE(isValidAsciiIdentifier("__a"));
  EXPECT_TRUE(isValidAsciiIdentifier("__z"));
  EXPECT_TRUE(isValidAsciiIdentifier("__A"));
  EXPECT_TRUE(isValidAsciiIdentifier("__Z"));
  EXPECT_TRUE(isValidAsciiIdentifier("__$", /*AllowDollar=*/true));

  // 3 characters, '_' prefix and suffix
  EXPECT_FALSE(isValidAsciiIdentifier("_._"));
  EXPECT_FALSE(isValidAsciiIdentifier("_\n_"));
  EXPECT_FALSE(isValidAsciiIdentifier("_ _"));
  EXPECT_FALSE(isValidAsciiIdentifier("_\x80_"));
  EXPECT_FALSE(isValidAsciiIdentifier("_\xc2_"));
  EXPECT_FALSE(isValidAsciiIdentifier("_\xff_"));
  EXPECT_FALSE(isValidAsciiIdentifier("_$_"));
  EXPECT_TRUE(isValidAsciiIdentifier("_1_"));

  EXPECT_TRUE(isValidAsciiIdentifier("___"));
  EXPECT_TRUE(isValidAsciiIdentifier("_a_"));
  EXPECT_TRUE(isValidAsciiIdentifier("_z_"));
  EXPECT_TRUE(isValidAsciiIdentifier("_A_"));
  EXPECT_TRUE(isValidAsciiIdentifier("_Z_"));
  EXPECT_TRUE(isValidAsciiIdentifier("_$_", /*AllowDollar=*/true));
}
