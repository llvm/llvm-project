//===-- DILLiteralParsers.Simple.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_VALUEOBJECT_DILLiteralParsers_H
#define LLDB_VALUEOBJECT_DILLiteralParsers_H

#include "lldb/ValueObject/DILCharInfo.h"
#include "lldb/ValueObject/DILLexer.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_private {

namespace dil {

class NumericLiteralParser {
 public:
  const char *const ThisTokBegin;
  const char *const ThisTokEnd;
  const char *DigitsBegin, *SuffixBegin; // markers
  const char *s; // cursor

  unsigned radix;
  bool saw_exponent, saw_period, saw_fixed_point_suffix;

  NumericLiteralParser(llvm::StringRef TokSpelling, unsigned TokLoc,
                       bool AllowHalfType,
                       DILLexer &lexer, bool AllowMicrosoftExt);

  bool hadError : 1;
  bool isUnsigned : 1;
  bool isLong : 1;          // This is *not* set for long long.
  bool isLongLong : 1;
  bool isSizeT : 1;         // 1z, 1uz (C++23)
  bool isHalf : 1;          // 1.0h
  bool isFloat : 1;         // 1.0f
  bool isImaginary : 1;     // 1.0i
  bool isFloat16 : 1;       // 1.0f16
  bool isFloat128 : 1;      // 1.0q
  bool isFract : 1;         // 1.0hr/r/lr/uhr/ur/ulr
  bool isAccum : 1;         // 1.0hk/k/lk/uhk/uk/ulk
  bool isBitInt : 1;        // 1wb, 1uwb (C23) or 1__wb, 1__uwb (Clang extension in C++
                            // mode)
  uint8_t MicrosoftInteger; // Microsoft suffix extension i8, i16, i32, or i64.


  bool isFixedPointLiteral() const {
    return (saw_period || saw_exponent) && saw_fixed_point_suffix;
  }

  bool isIntegerLiteral() const {
    return !saw_period && !saw_exponent && !isFixedPointLiteral();
  }
  bool isFloatingLiteral() const {
    return (saw_period || saw_exponent) && !isFixedPointLiteral();
  }

  unsigned getRadix() const { return radix; }

  /// GetIntegerValue - Convert this numeric literal value to an APInt that
  /// matches Val's input width.  If there is an overflow (i.e., if the unsigned
  /// value read is larger than the APInt's bits will hold), set Val to the low
  /// bits of the result and return true.  Otherwise, return false.
  bool GetIntegerValue(llvm::APInt &Val);

  /// Convert this numeric literal to a floating value, using the specified
  /// APFloat fltSemantics (specifying float, double, etc) and rounding mode.
  llvm::APFloat::opStatus GetFloatValue(llvm::APFloat &Result,
                                        llvm::RoundingMode RM);

  /// Get the digits that comprise the literal. This excludes any prefix or
  /// suffix associated with the literal.
  llvm::StringRef getLiteralDigits() const {
    assert(!hadError && "cannot reliably get the literal digits with an error");
    return llvm::StringRef(DigitsBegin, SuffixBegin - DigitsBegin);
  }

private:

  void ParseNumberStartingWithZero(unsigned TokLoc, bool AllowHexFloats);
  void ParseDecimalOrOctalCommon(unsigned TokLoc);

  static bool isDigitSeparator(char C) { return C == '\''; }

  /// Determine whether the sequence of characters [Start, End) contains
  /// any real digits (not digit separators).
  bool containsDigits(const char *Start, const char *End) {
    return Start != End && (Start + 1 != End || !isDigitSeparator(Start[0]));
  }

  enum CheckSeparatorKind { CSK_BeforeDigits, CSK_AfterDigits };

  /// Ensure that we don't have a digit separator here.
  void CheckSeparator(unsigned TokLoc, const char *Pos,
                      CheckSeparatorKind IsAfterDigits);

  /// SkipHexDigits - Read and skip over any hex digits, up to End.
  /// Return a pointer to the first non-hex digit or End.
  const char *SkipHexDigits(const char *ptr) {
    while (ptr != ThisTokEnd &&
           (isHexDigit(*ptr) || isDigitSeparator(*ptr)))
      ptr++;
    return ptr;
  }

  /// SkipOctalDigits - Read and skip over any octal digits, up to End.
  /// Return a pointer to the first non-hex digit or End.
  const char *SkipOctalDigits(const char *ptr) {
    while (ptr != ThisTokEnd &&
           ((*ptr >= '0' && *ptr <= '7') || isDigitSeparator(*ptr)))
      ptr++;
    return ptr;
  }

  /// SkipDigits - Read and skip over any digits, up to End.
  /// Return a pointer to the first non-hex digit or End.
  const char *SkipDigits(const char *ptr) {
    while (ptr != ThisTokEnd &&
           (isDigit(*ptr) || isDigitSeparator(*ptr)))
      ptr++;
    return ptr;
  }

  /// SkipBinaryDigits - Read and skip over any binary digits, up to End.
  /// Return a pointer to the first non-binary digit or End.
  const char *SkipBinaryDigits(const char *ptr) {
    while (ptr != ThisTokEnd &&
           (*ptr == '0' || *ptr == '1' || isDigitSeparator(*ptr)))
      ptr++;
    return ptr;
  }

};

class CharLiteralParser {
private:
  uint64_t Value;
  dil::TokenKind Kind;
  bool IsMultiChar;
  bool HadError;

public:
  CharLiteralParser(const char *begin, const char *end,
                    unsigned Loc, DILLexer &lexer,
                    dil::TokenKind kind);

  bool hadError() const { return HadError; }
  bool isOrdinary() const { return Kind == dil::TokenKind::char_constant; }
  bool isWide() const { return Kind == dil::TokenKind::wide_char_constant; }
  bool isUTF8() const { return Kind == dil::TokenKind::utf8_char_constant; }
  bool isMultiChar() const { return IsMultiChar; }
  uint64_t getValue() const { return Value; }
};

enum class StringLiteralEvalMethod {
  Evaluated,
  Unevaluated,
};

class StringLiteralParser {
  unsigned MaxTokenLength;
  unsigned SizeBound;
  unsigned CharByteWidth;
  dil::TokenKind Kind;
  llvm::SmallString<512> ResultBuf;
  char *ResultPtr; // cursor
  StringLiteralEvalMethod EvalMethod;
  DILLexer &m_lexer;

public:
  StringLiteralParser(llvm::ArrayRef<DILToken> StringToks,
                      DILLexer &lexer,
                      StringLiteralEvalMethod EvalMethod =
                      StringLiteralEvalMethod::Evaluated);

  bool hadError;

  llvm::StringRef GetString() const {
    return llvm::StringRef(ResultBuf.data(), GetStringLength());
  }
  unsigned GetStringLength() const { return ResultPtr-ResultBuf.data(); }

  unsigned GetNumStringChars() const {
    return GetStringLength() / CharByteWidth;
  }
  /// getOffsetOfStringByte - This function returns the offset of the
  /// specified byte of the string data represented by Token.  This handles
  /// advancing over escape sequences in the string.
  ///
  /// If the Diagnostics pointer is non-null, then this will do semantic
  /// checking of the string literal and emit errors and warnings.
  unsigned getOffsetOfStringByte(const DILToken &TheTok,
                                 unsigned ByteNo) const;

  bool isOrdinary() const { return Kind == dil::TokenKind::string_literal; }
  bool isWide() const { return Kind == dil::TokenKind::wide_string_literal; }
  bool isUTF8() const { return Kind == dil::TokenKind::utf8_string_literal; }
  bool isUnevaluated() const {
    return EvalMethod == StringLiteralEvalMethod::Unevaluated;
  }

private:
  void init(llvm::ArrayRef<DILToken> StringToks,
            DILLexer &lexer);
  bool CopyStringFragment(const DILToken &Tok, const char *TokBegin,
                          llvm::StringRef Fragment);
  void DiagnoseLexingError(unsigned Loc);
};

} // end namespace dil

} // end namespace lldb_private

#endif // LLDB_VALUEOBJECT_DILLiteralParsers_H
