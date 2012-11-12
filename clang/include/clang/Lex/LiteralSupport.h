//===--- LiteralSupport.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the NumericLiteralParser, CharLiteralParser, and
// StringLiteralParser interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LITERALSUPPORT_H
#define CLANG_LITERALSUPPORT_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"
#include "clang/Basic/TokenKinds.h"
#include <cctype>

namespace clang {

class DiagnosticsEngine;
class Preprocessor;
class Token;
class SourceLocation;
class TargetInfo;
class SourceManager;
class LangOptions;

/// NumericLiteralParser - This performs strict semantic analysis of the content
/// of a ppnumber, classifying it as either integer, floating, or erroneous,
/// determines the radix of the value and can convert it to a useful value.
class NumericLiteralParser {
  Preprocessor &PP; // needed for diagnostics

  const char *const ThisTokBegin;
  const char *const ThisTokEnd;
  const char *DigitsBegin, *SuffixBegin; // markers
  const char *s; // cursor

  unsigned radix;

  bool saw_exponent, saw_period, saw_ud_suffix;

public:
  NumericLiteralParser(StringRef TokSpelling,
                       SourceLocation TokLoc,
                       Preprocessor &PP);
  bool hadError;
  bool isUnsigned;
  bool isLong;        // This is *not* set for long long.
  bool isLongLong;
  bool isFloat;       // 1.0f
  bool isImaginary;   // 1.0i
  bool isMicrosoftInteger;  // Microsoft suffix extension i8, i16, i32, or i64.

  bool isIntegerLiteral() const {
    return !saw_period && !saw_exponent;
  }
  bool isFloatingLiteral() const {
    return saw_period || saw_exponent;
  }

  bool hasUDSuffix() const {
    return saw_ud_suffix;
  }
  StringRef getUDSuffix() const {
    assert(saw_ud_suffix);
    return StringRef(SuffixBegin, ThisTokEnd - SuffixBegin);
  }
  unsigned getUDSuffixOffset() const {
    assert(saw_ud_suffix);
    return SuffixBegin - ThisTokBegin;
  }

  unsigned getRadix() const { return radix; }

  /// GetIntegerValue - Convert this numeric literal value to an APInt that
  /// matches Val's input width.  If there is an overflow (i.e., if the unsigned
  /// value read is larger than the APInt's bits will hold), set Val to the low
  /// bits of the result and return true.  Otherwise, return false.
  bool GetIntegerValue(llvm::APInt &Val);

  /// GetFloatValue - Convert this numeric literal to a floating value, using
  /// the specified APFloat fltSemantics (specifying float, double, etc).
  /// The optional bool isExact (passed-by-reference) has its value
  /// set to true if the returned APFloat can represent the number in the
  /// literal exactly, and false otherwise.
  llvm::APFloat::opStatus GetFloatValue(llvm::APFloat &Result);

private:

  void ParseNumberStartingWithZero(SourceLocation TokLoc);

  /// SkipHexDigits - Read and skip over any hex digits, up to End.
  /// Return a pointer to the first non-hex digit or End.
  const char *SkipHexDigits(const char *ptr) {
    while (ptr != ThisTokEnd && isxdigit(*ptr))
      ptr++;
    return ptr;
  }

  /// SkipOctalDigits - Read and skip over any octal digits, up to End.
  /// Return a pointer to the first non-hex digit or End.
  const char *SkipOctalDigits(const char *ptr) {
    while (ptr != ThisTokEnd && ((*ptr >= '0') && (*ptr <= '7')))
      ptr++;
    return ptr;
  }

  /// SkipDigits - Read and skip over any digits, up to End.
  /// Return a pointer to the first non-hex digit or End.
  const char *SkipDigits(const char *ptr) {
    while (ptr != ThisTokEnd && isdigit(*ptr))
      ptr++;
    return ptr;
  }

  /// SkipBinaryDigits - Read and skip over any binary digits, up to End.
  /// Return a pointer to the first non-binary digit or End.
  const char *SkipBinaryDigits(const char *ptr) {
    while (ptr != ThisTokEnd && (*ptr == '0' || *ptr == '1'))
      ptr++;
    return ptr;
  }

};

/// CharLiteralParser - Perform interpretation and semantic analysis of a
/// character literal.
class CharLiteralParser {
  uint64_t Value;
  tok::TokenKind Kind;
  bool IsMultiChar;
  bool HadError;
  SmallString<32> UDSuffixBuf;
  unsigned UDSuffixOffset;
public:
  CharLiteralParser(const char *begin, const char *end,
                    SourceLocation Loc, Preprocessor &PP,
                    tok::TokenKind kind);

  bool hadError() const { return HadError; }
  bool isAscii() const { return Kind == tok::char_constant; }
  bool isWide() const { return Kind == tok::wide_char_constant; }
  bool isUTF16() const { return Kind == tok::utf16_char_constant; }
  bool isUTF32() const { return Kind == tok::utf32_char_constant; }
  bool isMultiChar() const { return IsMultiChar; }
  uint64_t getValue() const { return Value; }
  StringRef getUDSuffix() const { return UDSuffixBuf; }
  unsigned getUDSuffixOffset() const {
    assert(!UDSuffixBuf.empty() && "no ud-suffix");
    return UDSuffixOffset;
  }
};

/// StringLiteralParser - This decodes string escape characters and performs
/// wide string analysis and Translation Phase #6 (concatenation of string
/// literals) (C99 5.1.1.2p1).
class StringLiteralParser {
  const SourceManager &SM;
  const LangOptions &Features;
  const TargetInfo &Target;
  DiagnosticsEngine *Diags;
  
  unsigned MaxTokenLength;
  unsigned SizeBound;
  unsigned CharByteWidth;
  tok::TokenKind Kind;
  SmallString<512> ResultBuf;
  char *ResultPtr; // cursor
  SmallString<32> UDSuffixBuf;
  unsigned UDSuffixToken;
  unsigned UDSuffixOffset;
public:
  StringLiteralParser(const Token *StringToks, unsigned NumStringToks,
                      Preprocessor &PP, bool Complain = true);
  StringLiteralParser(const Token *StringToks, unsigned NumStringToks,
                      const SourceManager &sm, const LangOptions &features,
                      const TargetInfo &target, DiagnosticsEngine *diags = 0)
    : SM(sm), Features(features), Target(target), Diags(diags),
      MaxTokenLength(0), SizeBound(0), CharByteWidth(0), Kind(tok::unknown),
      ResultPtr(ResultBuf.data()), hadError(false), Pascal(false) {
    init(StringToks, NumStringToks);
  }
    

  bool hadError;
  bool Pascal;

  StringRef GetString() const {
    return StringRef(ResultBuf.data(), GetStringLength());
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
  unsigned getOffsetOfStringByte(const Token &TheTok, unsigned ByteNo) const;

  bool isAscii() const { return Kind == tok::string_literal; }
  bool isWide() const { return Kind == tok::wide_string_literal; }
  bool isUTF8() const { return Kind == tok::utf8_string_literal; }
  bool isUTF16() const { return Kind == tok::utf16_string_literal; }
  bool isUTF32() const { return Kind == tok::utf32_string_literal; }
  bool isPascal() const { return Pascal; }

  StringRef getUDSuffix() const { return UDSuffixBuf; }

  /// Get the index of a token containing a ud-suffix.
  unsigned getUDSuffixToken() const {
    assert(!UDSuffixBuf.empty() && "no ud-suffix");
    return UDSuffixToken;
  }
  /// Get the spelling offset of the first byte of the ud-suffix.
  unsigned getUDSuffixOffset() const {
    assert(!UDSuffixBuf.empty() && "no ud-suffix");
    return UDSuffixOffset;
  }

private:
  void init(const Token *StringToks, unsigned NumStringToks);
  bool CopyStringFragment(const Token &Tok, const char *TokBegin,
                          StringRef Fragment);
  void DiagnoseLexingError(SourceLocation Loc);
};

}  // end namespace clang

#endif
