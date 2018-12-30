//===- OclCxxDemangler.cpp - OCLC++ simple demangler            -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//
//
// Copyright (c) 2015 The Khronos Group Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
//
//===----------------------------------------------------------------------===//


#include "OclCxxDemangler.h"
#include "OclCxxParseVariant.h"

#include <cassert>
#include <limits>
#include <memory>
#include <stack>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace oclcxx::adaptation;


// -----------------------------------------------------------------------------
// TOKENIZER HELPERS
// -----------------------------------------------------------------------------

using ParsePosT = std::string::size_type;


/// \brief Matches fixed-size prefix.
template <std::size_t N>
inline static bool matchPrefix(const std::string& ParsedText,
                               std::string::size_type ParsePos,
                               const char (&Prefix)[N]) {
  if (ParsePos >= ParsedText.length())
    return false;
  return ParsedText.compare(ParsePos, N - 1, Prefix) == 0;
}

/// \brief Matches fixed-size prefix and advances position if succeeded.
template <std::size_t N>
inline static bool matchPrefixAndAdvance(const std::string& ParsedText,
                                         std::string::size_type& ParsePos,
                                         const char (&Prefix)[N]) {
  if (matchPrefix(ParsedText, ParsePos, Prefix)) {
    ParsePos += N - 1;
    return true;
  }
  return false;
}

// -----------------------------------------------------------------------------
// <number> ::= [n] /[0-9]|[1-9][0-9]+/

/// \brief Kind of number maching used by match function.
enum NumberMatchKind {
  NMK_Positive,     ///< Only positive integral numbers.
  NMK_NonNegative,  ///< Only non-negative integral numbers.
  NMK_NonPositive,  ///< Only non-positive integral numbers.
  NMK_Negative,     ///< Only negative intergal numbers.
  NMK_Any,          ///< Any integral number.
};

/// \brief Matches mangled decimal number and advances position if succeeded.
///
/// \tparam ResultT           Result type for matching.
/// \tparam MatchKind         Kind of decimal number match.
/// \tparam MatchLimit        Limit of decimal digits in match (0 - no limit).
/// \tparam AllowLeadingZeros Allows leading zeros in match.
/// \param ParsedText    Input text for match.
/// \param ParsePos      Start position of matching.
/// \param MatchedNumber Placeholder for matched number.
/// \return              Indicates that match succeeded.
template <typename ResultT, NumberMatchKind MatchKind = NMK_Any,
          ParsePosT MatchLimit = 0, bool AllowLeadingZeros = false>
inline static bool matchIntNumberAndAdvance(const std::string& ParsedText,
                                            ParsePosT& ParsePos,
                                            ResultT& MatchedNumber) {
  using UResultT = typename std::make_unsigned<ResultT>::type;
  const auto ResultTMin = std::numeric_limits<ResultT>::min();
  const auto ResultTMax = std::numeric_limits<ResultT>::max();
  const auto UResultTMax = std::numeric_limits<UResultT>::max();

  auto TextLen = ParsedText.length();
  auto SignPos = ParsePos;
  auto ResultSign = true; // Positive.
  UResultT Result = 0;

  if (SignPos < TextLen && ParsedText[SignPos] == 'n') {
    if (MatchKind == NMK_Positive || MatchKind == NMK_NonNegative)
      return false;
    assert(std::numeric_limits<ResultT>::is_signed &&
           "Matched number cannot be stored in result type (sign).");
    ResultSign = false;
    ++SignPos;
  }
  else if (MatchKind == NMK_Negative)
    return false;

  auto DigitPos = SignPos;
  while (DigitPos < TextLen &&
         (MatchLimit <= 0 || DigitPos < SignPos + MatchLimit)) {
    auto ParsedDigit = ParsedText[DigitPos] - '0';
    if (ParsedDigit < 0 || ParsedDigit > 9)
      break;
    // If we have valid next digit while first was 0 (deduced from accumulator),
    // we have leading zeros.
    if (!AllowLeadingZeros && Result == 0 && DigitPos > SignPos)
      return false;

    assert(Result <= UResultTMax / 10 &&
           Result * 10 + ParsedDigit >= Result * 10 &&
           "Matched number cannot be stored in result type.");
    Result = Result * 10 + ParsedDigit;
    ++DigitPos;
  }
  if (DigitPos == SignPos)
    return false;

  if ((MatchKind == NMK_Positive || MatchKind == NMK_Negative) && Result == 0)
    return false;
  if (MatchKind == NMK_NonPositive && ResultSign && Result > 0)
    return false;

  ParsePos = DigitPos;
  if (ResultSign) {
    assert(Result <= static_cast<UResultT>(ResultTMax) &&
           "Matched number cannot be stored in result.");
    MatchedNumber = static_cast<ResultT>(Result);
  }
  else {
#ifdef _MSC_VER
  // NOTE: This is for removal of annoying non-C++ compliant error when
  //       using negate operation with unsigned type. The behavior in C++11
  //       is well-defined, but on VS with SDL checks it will generate error
  //       here. This makes harder to use template code and we have already
  //       assert() which tests parsing signed with unsigned result, so we
  //       can disable this warning.
  #pragma warning(push)
  #pragma warning(disable: 4146)
#endif
    assert(Result <= static_cast<UResultT>(ResultTMin) &&
           "Matched number cannot be stored in result.");
    MatchedNumber = Result <= static_cast<UResultT>(ResultTMax)
                      ? -static_cast<ResultT>(Result)
                      : static_cast<ResultT>(Result); // 2's-complement assumed.
#ifdef _MSC_VER
  #pragma warning(pop)
#endif
  }
  return true;
}

// -----------------------------------------------------------------------------
// <source-name> ::= <number> <identifier> # number is positive
//
// <identifier> ::= [a-zA-Z0-9_\x80-\xFF]+ # sequence of bytes of UTF-8

/// \brief Matches mangled source name.
///
/// \param ParsedText        Input text for match.
/// \param ParsePos          Start position of matching.
/// \param MatchedIdentifier Placeholder for matched identifier.
/// \return                  Indicates that match succeeded.
inline static bool matchSourceNameAndAdvance(const std::string& ParsedText,
                                             ParsePosT& ParsePos,
                                             std::string& MatchedIdentifier) {
  auto TextLen = ParsedText.size();
  auto Pos = ParsePos;
  ParsePosT IdentifierSize = 0;

  if (!matchIntNumberAndAdvance<ParsePosT, NMK_Positive>(ParsedText, Pos,
                                                         IdentifierSize))
    return false;
  if (Pos + IdentifierSize < Pos || Pos + IdentifierSize > TextLen)
    return false;

  ParsePos = Pos + IdentifierSize;
  MatchedIdentifier = ParsedText.substr(Pos, IdentifierSize);
  return true;
}

// -----------------------------------------------------------------------------
// <seq-id> ::= [0-9A-Z]+

/// \brief Matches mangled sequential id and advances position if succeeded.
///
/// \tparam ResultT Result type for matching (unsigned integral recommended).
/// \param ParsedText    Input text for match.
/// \param ParsePos      Start position of matching.
/// \param MatchedSeqId  Placeholder for matched identifier.
/// \return              Indicates that match succeeded.
template <typename ResultT>
inline static bool matchSeqIdAndAdvance(const std::string& ParsedText,
                                        ParsePosT& ParsePos,
                                        ResultT& MatchedSeqId) {
  const auto ResultTMax = std::numeric_limits<ResultT>::max();

  auto TextLen = ParsedText.size();
  auto Pos = ParsePos;
  ResultT Result = 0;

  while (Pos < TextLen) {
    auto ParsedDigit = ParsedText[Pos]; // base-36 digit.
    if (ParsedDigit >= '0' && ParsedDigit <= '9')
      ParsedDigit -= '0';
    else if (ParsedDigit >= 'A' && ParsedDigit <= 'Z')
      ParsedDigit = ParsedDigit - 'A' + 10;
    else
      break;

    assert(Result <= ResultTMax / 36 &&
           Result * 36 + ParsedDigit >= Result * 36 &&
           "Matched sequence id cannot be stored in result type.");
    Result = Result * 36 + ParsedDigit;
    ++Pos;
  }
  if (Pos == ParsePos)
    return false;

  ParsePos = Pos;
  MatchedSeqId = Result;
  return true;
}

// -----------------------------------------------------------------------------
// <fixed-builtin-type> ::= v   # void
//                      ::= w   # wchar_t
//                      ::= b   # bool
//                      ::= c   # char
//                      ::= a   # signed char
//                      ::= h   # unsigned char
//                      ::= s   # short
//                      ::= t   # unsigned short
//                      ::= i   # int
//                      ::= j   # unsigned int
//                      ::= l   # long
//                      ::= m   # unsigned long
//                      ::= x   # long long, __int64
//                      ::= y   # unsigned long long, __int64
//                      ::= n   # __int128
//                      ::= o   # unsigned __int128
//                      ::= f   # float
//                      ::= d   # double
//                      ::= e   # long double, __float80
//                      ::= g   # __float128
//                      ::= z   # ellipsis
//                      ::= Dd  # IEEE 754r decimal floating point (64 bits)
//                      ::= De  # IEEE 754r decimal floating point (128 bits)
//                      ::= Df  # IEEE 754r decimal floating point (32 bits)
//                      ::= Dh  # IEEE 754r half-prec floating point (16 bits)
//                      ::= Di  # char32_t
//                      ::= Ds  # char16_t
//                      ::= Da  # auto
//                      ::= Dc  # decltype(auto)
//                      ::= Dn  # std::nullptr_t (i.e., decltype(nullptr))

/// C++ names for fixed built-in types (index corresponds DmngBuiltinType
/// enum value).
static const std::string FixedBuiltinTypeNames[] = {
#define OCLCXX_MENC_BITYPE_FIXED(name, encName, cxxName) cxxName,
#include "OclCxxMangleEncodings.inc"
#undef OCLCXX_MENC_BITYPE_FIXED
};

/// Itanium-encoded names for fixed built-in types (index corresponds
/// DmngBuiltinType enum value).
static const std::string EncFixedBuiltinTypeNames[] = {
#define OCLCXX_MENC_BITYPE_FIXED(name, encName, cxxName) encName,
#include "OclCxxMangleEncodings.inc"
#undef OCLCXX_MENC_BITYPE_FIXED
};


/// \brief Returns C++ name of built-in type.
const std::string &oclcxx::adaptation::getFixedBuiltinTypeName(
    DmngBuiltinType Type) {
  assert(Type >= 0 && Type < sizeof(FixedBuiltinTypeNames) /
                             sizeof(FixedBuiltinTypeNames[0])
                   && "Name of built-in type is not fixed name.");

  return FixedBuiltinTypeNames[Type];
}

/// \brief Returns Itanium-encoded name of built-in type.
const std::string &oclcxx::adaptation::getEncFixedBuiltinTypeName(
    DmngBuiltinType Type) {
  assert(Type >= 0 && Type < sizeof(EncFixedBuiltinTypeNames) /
                             sizeof(EncFixedBuiltinTypeNames[0])
                   && "Name of built-in type is not fixed name.");

  return EncFixedBuiltinTypeNames[Type];
}

/// \brief Matches built-in type and advances if succeeded.
static bool matchFixedBiTypeAndAdvance(const std::string &ParsedText,
                                       ParsePosT &ParsePos,
                                       DmngBuiltinType &MatchedType) {
  auto TextLen = ParsedText.length();
  auto Pos = ParsePos;
  if (Pos >= TextLen)
    return false;

  if (ParsedText[Pos] == 'D') {
    if (++Pos >= TextLen)
      return false;

    switch (ParsedText[Pos]) {
    case 'd': MatchedType = DBT_Float64R;
      break;
    case 'e': MatchedType = DBT_Float128R;
      break;
    case 'f': MatchedType = DBT_Float32R;
      break;
    case 'h': MatchedType = DBT_Half;
      break;
    case 'i': MatchedType = DBT_Char32;
      break;
    case 's': MatchedType = DBT_Char16;
      break;
    case 'a': MatchedType = DBT_Auto;
      break;
    case 'c': MatchedType = DBT_DecltypeAuto;
      break;
    case 'n': MatchedType = DBT_NullPtr;
      break;
    default:
      return false;
    }

    ParsePos = Pos + 1;
    return true;
  }

  switch (ParsedText[Pos]) {
  case 'v': MatchedType = DBT_Void;
    break;
  case 'w': MatchedType = DBT_WChar;
    break;
  case 'b': MatchedType = DBT_Bool;
    break;
  case 'c': MatchedType = DBT_Char;
    break;
  case 'a': MatchedType = DBT_SChar;
    break;
  case 'h': MatchedType = DBT_UChar;
    break;
  case 's': MatchedType = DBT_Short;
    break;
  case 't': MatchedType = DBT_UShort;
    break;
  case 'i': MatchedType = DBT_Int;
    break;
  case 'j': MatchedType = DBT_UInt;
    break;
  case 'l': MatchedType = DBT_Long;
    break;
  case 'm': MatchedType = DBT_ULong;
    break;
  case 'x': MatchedType = DBT_LongLong;
    break;
  case 'y': MatchedType = DBT_ULongLong;
    break;
  case 'n': MatchedType = DBT_Int128;
    break;
  case 'o': MatchedType = DBT_UInt128;
    break;
  case 'f': MatchedType = DBT_Float;
    break;
  case 'd': MatchedType = DBT_Double;
    break;
  case 'e': MatchedType = DBT_LongDouble;
    break;
  case 'g': MatchedType = DBT_Float128;
    break;
  case 'z': MatchedType = DBT_Ellipsis;
    break;
  default:
    return false;
  }

  ParsePos = Pos + 1;
  return true;
}

// -----------------------------------------------------------------------------
// <CV-qualifiers> ::= [r] [V] [K]   # restrict (C99), volatile, const
//
// Correct production:
// <CV-qualifiers> ::= r [V] [K]
//                 ::= [r] V [K]
//                 ::= [r] [V] K

/// \brief Matches cvr-qualifiers and advances position if succeeded.
inline static bool matchCvrQualsAndAdvance(const std::string &ParsedText,
                                           ParsePosT &ParsePos,
                                           DmngCvrQuals &MatchedQuals) {
  auto TextLen = ParsedText.length();
  auto Pos = ParsePos;
  auto Quals = DCVQ_None;

  if (Pos >= TextLen) {
    ParsePos = Pos;
    MatchedQuals = Quals;
    return false;
  }

  if (ParsedText[Pos] == 'r') {
    Quals = Quals | DCVQ_Restrict;
    if (++Pos >= TextLen) {
      ParsePos = Pos;
      MatchedQuals = Quals;
      return true;
    }
  }

  if (ParsedText[Pos] == 'V') {
    Quals = Quals | DCVQ_Volatile;
    if (++Pos >= TextLen) {
      ParsePos = Pos;
      MatchedQuals = Quals;
      return true;
    }
  }

  if (ParsedText[Pos] == 'K') {
    Quals = Quals | DCVQ_Const;
    ParsePos = Pos + 1;
    MatchedQuals = Quals;
    return true;
  }

  return false;
}

// -----------------------------------------------------------------------------
// <ref-qualifier> ::= R # & ref-qualifier
//                 ::= O # && ref-qualifier

/// \brief Matches ref-qualifiers and advances position if succeeded.
inline static bool matchRefQualsAndAdvance(const std::string &ParsedText,
                                           ParsePosT &ParsePos,
                                           DmngRefQuals &MatchedQuals) {
  auto TextLen = ParsedText.length();
  auto Pos = ParsePos;
  if (Pos >= TextLen)
    return false;

  switch(ParsedText[Pos]) {
  case 'R': MatchedQuals = DRQ_LValueRef;
    break;
  case 'O': MatchedQuals = DRQ_RValueRef;
    break;
  default:
    return false;
  }

  ParsePos = Pos + 1;
  return true;
}

// -----------------------------------------------------------------------------
// <fixed-operator-name> ::= nw   # new
//                       ::= na   # new[]
//                       ::= dl   # delete
//                       ::= da   # delete[]
//                       ::= ps   # + (unary)
//                       ::= ng   # - (unary)
//                       ::= ad   # & (unary)
//                       ::= de   # * (unary)
//                       ::= co   # ~
//                       ::= pl   # +
//                       ::= mi   # -
//                       ::= ml   # *
//                       ::= dv   # /
//                       ::= rm   # %
//                       ::= an   # &
//                       ::= or   # |
//                       ::= eo   # ^
//                       ::= aS   # =
//                       ::= pL   # +=
//                       ::= mI   # -=
//                       ::= mL   # *=
//                       ::= dV   # /=
//                       ::= rM   # %=
//                       ::= aN   # &=
//                       ::= oR   # |=
//                       ::= eO   # ^=
//                       ::= ls   # <<
//                       ::= rs   # >>
//                       ::= lS   # <<=
//                       ::= rS   # >>=
//                       ::= eq   # ==
//                       ::= ne   # !=
//                       ::= lt   # <
//                       ::= gt   # >
//                       ::= le   # <=
//                       ::= ge   # >=
//                       ::= nt   # !
//                       ::= aa   # &&
//                       ::= oo   # ||
//                       ::= pp   # ++ (postfix in <expression> context)
//                       ::= mm   # -- (postfix in <expression> context)
//                       ::= cm   # ,
//                       ::= pm   # ->*
//                       ::= pt   # ->
//                       ::= cl   # ()
//                       ::= ix   # []
//                       ::= qu   # ?

/*
NOTE: Cases created with simple PS script:

$a = @("nw", "na", "dl", "da", "ps", "ng", "ad", "de", "co", "pl", "mi", "ml",
       "dv", "rm", "an", "or", "eo", "aS", "pL", "mI", "mL", "dV", "rM", "aN",
       "oR", "eO", "ls", "rs", "lS", "rS", "eq", "ne", "lt", "gt", "le", "ge",
       "nt", "aa", "oo", "pp", "mm", "cm", "pm", "pt", "cl", "ix", "qu")

[string]::join("`n", @($a | % { $_[0] } | sort -Unique | % {
  $c = $_;
  $sc = [string]::join("`n", @($a | ?{ $_[0] -eq $c} | sort | % {
    "    case '{0}':`n      break;" -f  $_[1]
  }));
  "case '{0}':`n  if (++Pos >= TextLen)`n    return false;`n" +
  "  switch(ParsedText[Pos]) {{`n{1}`n    default:`n      return false;`n  }}" +
  "`n  break;" -f ($c, $sc)
}))
*/

/// C++ names for fixed operator names (index corresponds DmngOperatorName
/// enum value).
static const std::string FixedOperatorNames[] = {
#define OCLCXX_MENC_OPR_FIXED(name, encName, arity, cxxName) cxxName,
#include "OclCxxMangleEncodings.inc"
#undef OCLCXX_MENC_OPR_FIXED
};

/// Itanium-encoded names for fixed operator names (index corresponds
/// DmngOperatorName enum value).
static const std::string EncFixedOperatorNames[] = {
#define OCLCXX_MENC_OPR_FIXED(name, encName, arity, cxxName) encName,
#include "OclCxxMangleEncodings.inc"
#undef OCLCXX_MENC_OPR_FIXED
};

static const int InExprOperatorFixedArities[] = {
#define OCLCXX_MENC_OPR_FIXED(name, encName, arity, cxxName) arity,
#include "OclCxxMangleEncodings.inc"
#undef OCLCXX_MENC_OPR_FIXED

#define OCLCXX_MENC_OPR(name, arity) arity,
#include "OclCxxMangleEncodings.inc"
#undef OCLCXX_MENC_OPR
};


/// \brief Returns C++ name of operator (without "operator" prefix).
const std::string &oclcxx::adaptation::getFixedOperatorName(
    DmngOperatorName NameCode) {
  assert(NameCode >= 0 && NameCode < sizeof(FixedOperatorNames) /
                                     sizeof(FixedOperatorNames[0])
                       && "Operator name is not fixed name.");

  return FixedOperatorNames[NameCode];
}

/// \brief Returns Itanium-encoded name of operator.
const std::string &oclcxx::adaptation::getEncFixedOperatorName(
    DmngOperatorName NameCode) {
  assert(NameCode >= 0 && NameCode < sizeof(EncFixedOperatorNames) /
                                     sizeof(EncFixedOperatorNames[0])
                       && "Operator name is not fixed name.");

  return EncFixedOperatorNames[NameCode];
}

/// \brief Returns fixed arity of operator in <expression> context.
///
/// \return Arity of operator, or 0 if arity is unknown, operator requires
///         special form in <expression> or number of operands is variable.
int oclcxx::adaptation::getInExprOperatorFixedArity(DmngOperatorName NameCode) {
  assert(NameCode >= 0 && NameCode < sizeof(InExprOperatorFixedArities) /
                                     sizeof(InExprOperatorFixedArities[0])
                       && "Operator name is invalid / has no arity defined.");

  return InExprOperatorFixedArities[NameCode];
}

/// \brief Matches operator name and advances if succeeded.
///
/// \tparam InExpression Removes operators from matching which require special
///         treatment in expressions.
template <bool InExpression = false>
static bool matchFixedOperatorAndAdvance(const std::string &ParsedText,
                                         ParsePosT &ParsePos,
                                         DmngOperatorName &MatchedOperator) {
  auto TextLen = ParsedText.length();
  auto Pos = ParsePos;
  if (Pos >= TextLen)
    return false;

  switch (ParsedText[Pos]) {
  case 'a':
    if (++Pos >= TextLen)
      return false;
    switch (ParsedText[Pos]) {
    case 'a': MatchedOperator = DON_LogicalAnd;
      break;
    case 'd': MatchedOperator = DON_AddressOf;
      break;
    case 'N': MatchedOperator = DON_BitwiseAndAssign;
      break;
    case 'n': MatchedOperator = DON_BitwiseAnd;
      break;
    case 'S': MatchedOperator = DON_Assign;
      break;
    default:
      return false;
    }
    break;
  case 'c':
    if (++Pos >= TextLen)
      return false;
    switch (ParsedText[Pos]) {
    case 'l':
      if (InExpression)
        return false;
      MatchedOperator = DON_Call;
      break;
    case 'm': MatchedOperator = DON_Comma;
      break;
    case 'o': MatchedOperator = DON_BitwiseComplement;
      break;
    default:
      return false;
    }
    break;
  case 'd':
    if (++Pos >= TextLen)
      return false;
    switch (ParsedText[Pos]) {
    case 'a':
      if (InExpression)
        return false;
      MatchedOperator = DON_DeleteArray;
      break;
    case 'e': MatchedOperator = DON_Dereference;
      break;
    case 'l':
      if (InExpression)
        return false;
      MatchedOperator = DON_Delete;
      break;
    case 'V': MatchedOperator = DON_DivideAssign;
      break;
    case 'v': MatchedOperator = DON_Divide;
      break;
    default:
      return false;
    }
    break;
  case 'e':
    if (++Pos >= TextLen)
      return false;
    switch (ParsedText[Pos]) {
    case 'O': MatchedOperator = DON_BitwiseExclusiveOrAssign;
      break;
    case 'o': MatchedOperator = DON_BitwiseExclusiveOr;
      break;
    case 'q': MatchedOperator = DON_Equal;
      break;
    default:
      return false;
    }
    break;
  case 'g':
    if (++Pos >= TextLen)
      return false;
    switch (ParsedText[Pos]) {
    case 'e': MatchedOperator = DON_GreaterEqual;
      break;
    case 't': MatchedOperator = DON_GreaterThan;
      break;
    default:
      return false;
    }
    break;
  case 'i':
    if (++Pos >= TextLen)
      return false;
    switch (ParsedText[Pos]) {
    case 'x': MatchedOperator = DON_Index;
      break;
    default:
      return false;
    }
    break;
  case 'l':
    if (++Pos >= TextLen)
      return false;
    switch (ParsedText[Pos]) {
    case 'e': MatchedOperator = DON_LessEqual;
      break;
    case 's': MatchedOperator = DON_LeftShift;
      break;
    case 'S': MatchedOperator = DON_LeftShiftAssign;
      break;
    case 't': MatchedOperator = DON_LessThan;
      break;
    default:
      return false;
    }
    break;
  case 'm':
    if (++Pos >= TextLen)
      return false;
    switch (ParsedText[Pos]) {
    case 'I': MatchedOperator = DON_MinusAssign;
      break;
    case 'i': MatchedOperator = DON_Minus;
      break;
    case 'L': MatchedOperator = DON_MultiplyAssign;
      break;
    case 'l': MatchedOperator = DON_Multiply;
      break;
    case 'm':
      if (InExpression)
        return false;
      MatchedOperator = DON_MinusMinus;
      break;
    default:
      return false;
    }
    break;
  case 'n':
    if (++Pos >= TextLen)
      return false;
    switch (ParsedText[Pos]) {
    case 'a':
      if (InExpression)
        return false;
      MatchedOperator = DON_NewArray;
      break;
    case 'e': MatchedOperator = DON_NotEqual;
      break;
    case 'g': MatchedOperator = DON_Negative;
      break;
    case 't': MatchedOperator = DON_LogicalNegate;
      break;
    case 'w':
      if (InExpression)
        return false;
      MatchedOperator = DON_New;
      break;
    default:
      return false;
    }
    break;
  case 'o':
    if (++Pos >= TextLen)
      return false;
    switch (ParsedText[Pos]) {
    case 'o': MatchedOperator = DON_LogicalOr;
      break;
    case 'r': MatchedOperator = DON_BitwiseOr;
      break;
    case 'R': MatchedOperator = DON_BitwiseOrAssign;
      break;
    default:
      return false;
    }
    break;
  case 'p':
    if (++Pos >= TextLen)
      return false;
    switch (ParsedText[Pos]) {
    case 'l': MatchedOperator = DON_Plus;
      break;
    case 'L': MatchedOperator = DON_PlusAssign;
      break;
    case 'm': MatchedOperator = DON_PointerToMemberAccess;
      break;
    case 'p':
      if (InExpression)
        return false;
      MatchedOperator = DON_PlusPlus;
      break;
    case 's': MatchedOperator = DON_Positive;
      break;
    case 't': MatchedOperator = DON_MemberAccess;
      break;
    default:
      return false;
    }
    break;
  case 'q':
    if (++Pos >= TextLen)
      return false;
    switch (ParsedText[Pos]) {
    case 'u': MatchedOperator = DON_Conditional;
      break;
    default:
      return false;
    }
    break;
  case 'r':
    if (++Pos >= TextLen)
      return false;
    switch (ParsedText[Pos]) {
    case 'M': MatchedOperator = DON_RemainderAssign;
      break;
    case 'm': MatchedOperator = DON_Remainder;
      break;
    case 'S': MatchedOperator = DON_RightShiftAssign;
      break;
    case 's': MatchedOperator = DON_RightShift;
      break;
    default:
      return false;
    }
    break;
  default:
    return false;
  }

  ParsePos = Pos + 1;
  return true;
}

// -----------------------------------------------------------------------------
// <ctor-dtor-name> ::= C1   # complete object constructor
//                  ::= C2   # base object constructor
//                  ::= C3   # complete object allocating constructor
//                  ::= D0   # deleting destructor
//                  ::= D1   # complete object destructor
//                  ::= D2   # base object destructor

/// \brief Matches constructor or destructor name and advances if succeeded.
inline static bool matchCtorDtorAndAdvance(const std::string &ParsedText,
                                           ParsePosT &ParsePos,
                                           bool & IsCtor,
                                           DmngCtorDtorType &CtorDtorType) {
  auto TextLen = ParsedText.length();
  auto Pos = ParsePos;
  if (Pos >= TextLen)
    return false;

  if (ParsedText[Pos] == 'C') {
    if (++Pos >= TextLen)
      return false;

    switch (ParsedText[Pos]) {
    case '1': CtorDtorType = DCDT_CompleteObj;
      break;
    case '2': CtorDtorType = DCDT_BaseObj;
      break;
    case '3': CtorDtorType = DCDT_DynMemObj;
      break;
    default:
      return false;
    }

    ParsePos = Pos + 1;
    IsCtor = true;
    return true;
  }

  if (ParsedText[Pos] == 'D') {
    if (++Pos >= TextLen)
      return false;

    switch (ParsedText[Pos]) {
    case '0': CtorDtorType = DCDT_DynMemObj;
      break;
    case '1': CtorDtorType = DCDT_CompleteObj;
      break;
    case '2': CtorDtorType = DCDT_BaseObj;
      break;
    default:
      return false;
    }

    ParsePos = Pos + 1;
    IsCtor = false;
    return true;
  }

  return false;
}

// -----------------------------------------------------------------------------
// <std-sub> ::= St   # ::std
//           ::= Sa   # ::std::allocator
//           ::= Sb   # ::std::basic_string
//           ::= Ss   # ::std::basic_string <char,
//                    #   ::std::char_traits<char>, ::std::allocator<char>>
//           ::= Si   # ::std::basic_istream<char, std::char_traits<char>>
//           ::= So   # ::std::basic_ostream<char, std::char_traits<char>>
//           ::= Sd   # ::std::basic_iostream<char, std::char_traits<char>>

/// \brief Matches standard substitution identifier and advances if succeeded.
inline static bool matchStdSubAndAdvance(const std::string &ParsedText,
                                         ParsePosT &ParsePos,
                                         std::string &SubId) {
  auto TextLen = ParsedText.length();
  auto Pos = ParsePos;
  if (Pos >= TextLen)
    return false;

  if (ParsedText[Pos] != 'S' || ++Pos >= TextLen)
      return false;

  switch (ParsedText[Pos]) {
    case 't':
    case 'a':
    case 'b':
    case 's':
    case 'i':
    case 'o':
    case 'd':
      SubId = ParsedText.substr(ParsePos, 2);
      break;
    default:
      return false;
  }

  ParsePos = Pos + 1;
  return true;
}


// -----------------------------------------------------------------------------
// RESULT NODES ADAPTERS AND CONVERTERS
// -----------------------------------------------------------------------------

/// \brief Adapts node to specific node kind.
///
/// \param Node Node to adapt.
/// \return     Adapted node, or empty shared pointer if node cannot be adapted.
template <typename TargetType>
static std::shared_ptr<TargetType> adaptNode(
    const std::shared_ptr<DmngRsltNode> &Node) {
  static_assert(AlwaysFalse<TargetType>::value,
                "adaptNode() cannot adapt to selected TargetType.");
  return nullptr;
}

template <>
std::shared_ptr<DmngRsltType> adaptNode<DmngRsltType>(
    const std::shared_ptr<DmngRsltNode> &Node) {
  if (Node == nullptr)
    return nullptr;

  switch (Node->getNodeKind()) {
  case DNDK_Name: {
      auto &&NdName = Node->getAs<DNDK_Name>();
      if (NdName->getKind() == DNK_Ordinary) {
        return std::make_shared<DmngRsltTypeNameType>(
          NdName->getAs<DNK_Ordinary>());
      }
      return nullptr;
    }
  case DNDK_NamePart:
    return std::make_shared<DmngRsltTypeNameType>(
      std::make_shared<DmngRsltOrdinaryName>(Node->getAs<DNDK_NamePart>()));
  case DNDK_Type:
    return Node->getAs<DNDK_Type>();
  case DNDK_Expr: {
      auto &&NdExpr = Node->getAs<DNDK_Expr>();
      if (NdExpr->getKind() == DXK_TemplateParam) {
        return std::make_shared<DmngRsltTParamType>(
          NdExpr->getAs<DXK_TemplateParam>());
      }
      if (NdExpr->getKind() == DXK_Decltype) {
        return std::make_shared<DmngRsltDecltypeType>(
          NdExpr->getAs<DXK_Decltype>());
      }
      // ReSharper disable once CppUnreachableCode
      assert(false && "Other expressions should not be substitutions.");
      return nullptr;
    }
  case DNDK_NameParts:
    return std::make_shared<DmngRsltTypeNameType>(
      std::make_shared<DmngRsltOrdinaryName>(Node->getAs<DNDK_NameParts>()));
  default:
    return nullptr;
  }
}

template <>
std::shared_ptr<DmngRsltNameParts> adaptNode<DmngRsltNameParts>(
    const std::shared_ptr<DmngRsltNode> &Node) {
  if (Node == nullptr)
    return nullptr;

  switch (Node->getNodeKind()) {
  case DNDK_Name:
    return nullptr;
  case DNDK_NamePart:
    return std::make_shared<DmngRsltNameParts>(Node->getAs<DNDK_NamePart>());
  case DNDK_Type:
    return nullptr;
  case DNDK_Expr: {
      auto &&NdExpr = Node->getAs<DNDK_Expr>();
      if (NdExpr->getKind() == DXK_TemplateParam) {
        return std::make_shared<DmngRsltNameParts>(
          std::make_shared<DmngRsltTParamNamePart>(
            NdExpr->getAs<DXK_TemplateParam>()));
      }
      if (NdExpr->getKind() == DXK_Decltype) {
        return std::make_shared<DmngRsltNameParts>(
          std::make_shared<DmngRsltDecltypeNamePart>(
            NdExpr->getAs<DXK_Decltype>()));
      }
      // ReSharper disable once CppUnreachableCode
      assert(false && "Other expressions should not be substitutions.");
      return nullptr;
    }
  case DNDK_NameParts:
    return Node->getAs<DNDK_NameParts>();
  default:
    return nullptr;
  }
}


/// \brief Creates clone of template parameter expression based on
///        node which contains a possibly wrapped template parameter expression.
///
/// \param Node Node with possibly wrapped template parameter expression.
/// \return     Cloned expression node, or empty shared pointer if Node
///             does not contain the expression.
static std::shared_ptr<DmngRsltTParamExpr> createUnwrappedTParamExpr(
    const std::shared_ptr<const DmngRsltNode> &Node) {
  if (Node == nullptr)
    return nullptr;

  switch (Node->getNodeKind()) {
  case DNDK_Name:
    return nullptr;
  case DNDK_NamePart: {
      const auto &NdTParamNPart =
        Node->getAs<DNDK_NamePart>()->getAs<DNPK_TemplateParam>();
      if (NdTParamNPart == nullptr)
        return nullptr;
      return std::static_pointer_cast<DmngRsltTParamExpr>(
        NdTParamNPart->getTemplateParam()->clone());
    }
  case DNDK_Type: {
      const auto &NdTParamType =
        Node->getAs<DNDK_Type>()->getAs<DTK_TemplateParam>();
      if (NdTParamType == nullptr)
        return nullptr;
      return std::static_pointer_cast<DmngRsltTParamExpr>(
        NdTParamType->getTemplateParam()->clone());
    }
  case DNDK_Expr:
      return Node->clone()->getAs<DNDK_Expr>()->getAs<DXK_TemplateParam>();
  case DNDK_NameParts: {
      const auto &NdNParts = Node->getAs<DNDK_NameParts>();
      if (NdNParts->getParts().size() != 1)
        return nullptr;
      const auto &NdNPart = NdNParts->getParts()[0];
      if (NdNPart == nullptr)
        return nullptr;
      const auto &NdTParamNPart = NdNPart->getAs<DNPK_TemplateParam>();
      if (NdTParamNPart == nullptr)
          return nullptr;
        return std::static_pointer_cast<DmngRsltTParamExpr>(
          NdTParamNPart->getTemplateParam()->clone());
    }
  default:
    return nullptr;
  }
}


// -----------------------------------------------------------------------------
// PARSING CONTEXT AND STATE
// -----------------------------------------------------------------------------

namespace {

/// \brief Parsing conext (forward-declaration).
struct ParseContext;

/// \brief Identifier of parse state.
enum ParseStateId {
  PSI_Start,
  PSI_MangledName,
  PSI_Encoding,
  PSI_Name,
  PSI_BareFunctionType,
  PSI_SpecialName,
  PSI_NestedName,
  PSI_UnscopedName,
  PSI_UnscopedTemplateName,
  PSI_TemplateArgs,
  PSI_LocalName,
  PSI_Type,
  PSI_CallOffset,
  PSI_VendorQuals,
  PSI_UnqualifiedName,
  PSI_Substitution,
  PSI_TemplateArg,
  PSI_Discriminator,
  PSI_BuiltinType,
  PSI_FunctionType,
  PSI_ClassEnumType,
  PSI_ArrayType,
  PSI_VectorType,
  PSI_PointerToMemberType,
  PSI_TemplateParam,
  PSI_TemplateTemplateParam,
  PSI_Decltype,
  PSI_VendorQual,
  PSI_DataMemberPrefix,
  PSI_UnnamedTypeName,
  PSI_OperatorName,
  PSI_Expression,
  PSI_ExprPrimary,

  // Helper states (to minimize reparses).
  PSI_MangledName_Bug,           // For bug in clang 3.6 mangling.
  PSI_NestedName_Sfx,
  PSI_LocalName_Enty,
  PSI_NestedName_Start,
  PSI_NestedName_Step,
  PSI_TemplateArg_Pack,          // Different action -> do not add to TArgs.
};

/// \brief Parse state in simple demangling parser.
struct ParseState {
  /// Type of scope for template parameter resolution.
  ///
  /// The entry contains triple - boolean value and pair of indices:
  /// First boolean value indicates that the template argument scope requires
  /// late-referencing of template arguments.
  /// The second value: index that identifies parent scope,
  /// or -1 (converted to unsigned size type) if the scope has no parent scope;
  /// The third value: index that identifies argument group referred in selected
  /// scope, or -1 (converted to unsigned size type) if scope does not have
  /// argument group (yet).
  using TArgScopeT = std::tuple<bool, std::size_t, std::size_t>;


  /// \brief Advances dot for one position (without moving parse position).
  void advance() {
    ++DotPos;
    DotSuccess = true;
  }

  /// \brief Advances dot for one position (with moving parse position).
  void advance(ParsePosT NewPos) {
    assert(NewPos >= StartPos &&
           "Advance position move parse position before start position.");
    Pos = NewPos;
    advance();
  }

  /// \brief Moves dot to position of next production and optionally resets
  ///        parse position to start position for state.
  void moveToProduction(unsigned ProdDotPos, bool ResetPosToStartPos = true) {
    assert(ProdDotPos >= DotPos &&
          "Move must advance checked dot position to next production.");
    DotPos = ProdDotPos;
    DotSuccess = true;
    if (ResetPosToStartPos) {
      Pos = StartPos;
      resetContext();
    }
  }

  /// \brief Resets state dot and (optionally) parse position.
  void reset(bool ResetPosToStartPos = true) {
    DotPos = 0;
    DotSuccess = true;
    if (ResetPosToStartPos) {
      Pos = StartPos;
      resetContext();
    }
  }

  /// \brief Mark state as failed (try parse).
  void fail() {
    DotSuccess = false;
  }


  /// \brief Resets variables of parse context to the state from the start of
  ///        current state (simplified).
  ///
  /// It resizes all collections to state from start of current parse state
  /// (if possible). This solution works only correctly if no elements
  /// are removed beyond size when parse state was created.
  void resetContext();


  /// \brief Adds substitution for current state.
  ///
  /// It uses current state's parsing positions (start and current) to
  /// create identifier of substitution.
  ///
  /// \param SubRepr Representation for substitution. It will be cloned to
  ///                collection of substitutions.
  void addSubstitute(const std::shared_ptr<const DmngRsltNode> &SubRepr);

  /// \brief Adds substitution for current state (by move).
  ///
  /// It uses current state's parsing positions (start and current) to
  /// create identifier of substitution.
  ///
  /// \param SubRepr Representation for substitution. It will be moved to
  ///                collection of substitutions (it will NOT BE cloned).
  void addSubstituteNoClone(std::shared_ptr<DmngRsltNode> &&SubRepr);


  /// \brief Adds new group of template arguments.
  ///
  /// The method must be called before calling any addTArgToGroup or
  /// assignTArgGroupToScope in current state or in any substates.
  void addTArgGroup();

  /// \brief Adds template argument to current argument group (by copy).
  ///
  /// The method should only be called after addTArgGroup in current or
  /// ancestor state.
  ///
  /// \param Arg Template argument to add.
  void addTArg(const std::shared_ptr<DmngRsltTArg> &Arg);

  /// \brief Adds template argument to current argument group (by move).
  ///
  /// The method should only be called after addTArgGroup in current or
  /// ancestor state.
  ///
  /// \param Arg Template argument to add.
  void addTArg(std::shared_ptr<DmngRsltTArg> &&Arg);

  /// \brief Indicates that state has created its own scope of template
  ///        arguments.
  ///
  /// \return true if state has its own scope; otherwise, false.
  bool hasOwnTArgScope() const {
    return TArgScopeIdx != TArgScopeStartIdx;
  }

  /// \brief Adds new scope for template arguments (for current state and
  ///        substates).
  ///
  /// The method must be called before calling any assignTArgGroupToScope
  /// in current state or in any substates.
  ///
  /// The method creates new scope for template arguments for current state and
  /// its children. The scope defines range in which assigned template argument
  /// group can be referred from template parameters existing in that scope.
  /// Subscopes that do not have group assigned by default inherit referred
  /// group from parent. If they have group assigned it will hide selection
  /// from parent until end of subscope.
  /// Normally the assignment of new group into scope will affect only template
  /// parameters added after this assignment, but when scope with
  /// late-referencing of template parameters is created, the assignment of new
  /// group will affect all template parameters in that scope.
  ///
  /// \param LateReferencing Indicates that scope requires late referencing
  ///                        of template arguments (to partially support
  ///                        bizzare case of convesion operator).
  void addTArgScope(bool LateReferencing = false);

  /// \brief Changes behavior of referencing inside current scope.
  ///
  /// Only template parameters added after change are affected.
  ///
  /// \param LateReferencing Indicates that scope requires late referencing
  ///                        of template arguments (to partially support
  ///                        bizzare case of convesion operator).
  void setTArgScope(bool LateReferencing = false);

  /// \brief Assigns current template argument group to current scope.
  ///
  /// Both argument group and argument scope must be created before use of
  /// this method.
  void assignTArgGroupToScope();

  /// \brief Prepares referencing context for template paramater.
  ///
  /// \return Pair with values needed to identify referencing context
  ///         for template parameter (boolean value indicating that
  ///         late-referencing should be done and index of scope or group
  ///         (depending on late-referencing flag).
  std::pair<bool, std::size_t> getTParamReferencingContext() const;

  /// \brief Adds template parameter to current scope (by copy).
  ///
  /// The method should only be called after addTArgScope in current or
  /// ancestor state.
  ///
  /// \param Param Template parameter to add (for referencing).
  void addTParam(const std::shared_ptr<DmngRsltTParamExpr> &Param);

  /// \brief Adds template parameter to current scope (by move).
  ///
  /// The method should only be called after addTArgScope in current or
  /// ancestor state.
  ///
  /// \param Param Template parameter to add (for referencing).
  void addTParam(std::shared_ptr<DmngRsltTParamExpr> &&Param);


  /// \brief Creates new parse state.
  explicit ParseState(ParseContext &Ctx, ParseStateId Id, ParsePosT Pos = 0,
                      std::size_t TArgGrpIdx = static_cast<std::size_t>(-1),
                      std::size_t TArgScopeIdx = static_cast<std::size_t>(-1),
                      bool GatherSubsEnabled = true);


  /// Parse context.
  ParseContext &Ctx;

  /// Position in parsed text at start of parsing of current non-terminal.
  const ParsePosT StartPos;
  /// Current position in parsed text.
  ParsePosT Pos;

  /// State identifier (currently checked non-terminal).
  const ParseStateId Id;

  /// Currently checked position in productions of current non-terminal.
  /// (start dot position is always 0).
  unsigned DotPos;
  /// Indicates that parsing of last (non-)terminal was successful and
  /// checked position in productions changed.
  bool DotSuccess;

  /// Indicates that substition gathering is allowed in current context.
  bool GatherSubsEnabled;

  // Functionality for resetting parse position.

  /// Size of Substitutions collection at start of parsing current non-terminal.
  const std::size_t SubsSize;
  /// Size of TParams collection at start of parsing current non-terminal.
  const std::size_t TParamsSize;
  /// Size of TArgs collection at start of parsing current non-terminal.
  const std::size_t TArgsSize;
  /// Size of argument group at start of parsing current non-terminal (used to
  /// reset position in cooperation with TArgCurGrpStartIdx).
  const std::size_t TArgCurGrpSize;
  /// Size of TArgScopes collection at start of parsing current non-terminal.
  const std::size_t TArgScopesSize;
  /// Initial value of scope at start of parsing current non-terminal (used to
  /// reset scope in cooperation with TArgScopeStartIdx).
  const TArgScopeT TArgScopeStartVal;
  /// Size of Variables collection at start of parsing current non-terminal.
  const std::size_t VarsSize;

  // Functionality for referencing template arguments in nested scopes.

  /// Current template argument group (currently extended) at start of current
  /// non-terminal.
  const std::size_t TArgCurGrpStartIdx;
  /// Current template argument group (currently extended).
  std::size_t TArgCurGrpIdx;

  /// Current template argument scope index (for template parameters) at start
  /// of current non-terminal.
  const std::size_t TArgScopeStartIdx;
  /// Current template argument scope index (for template parameters).
  std::size_t TArgScopeIdx;
};

/// Parse context for parser (demangler).
struct ParseContext {
  /// Type of collection with parse states.
  using StatesT = std::stack<ParseState>;

  /// Type of substitutions collection.
  ///
  /// Pair represents part of mangled name and corresponding demangled name
  /// node that represent the substitution in mangling.
  using SubsT = std::vector<std::pair<std::string,
                                      std::shared_ptr<DmngRsltNode>>>;

  /// Type of helper set which contains parts of mangled names that allow to
  /// quickly detect and elimnate duplicates in substitutions.
  using SubsStrsT = std::unordered_set<std::string>;

  /// Type of collection with all template parameters (for late referencing).
  ///
  /// The collection contains triplets:
  /// First boolean value indicates that the template parameter will be
  /// late-referenced to template argument.
  /// The second value is an index for either template argument group (when
  /// first value is false / no late-referencing) or for template argument scope
  /// (when first value is true / late-referencing).
  /// The last value identifies result node of template parameter expression
  /// which will be linked to proper template argument node.
  using TParamsT = std::vector<std::tuple<bool,
                                          std::size_t,
                                          std::shared_ptr<DmngRsltTParamExpr>>>;

  /// Type of collection with all template arguments (for late referencing).
  ///
  /// Parameters are resolved in later phase to specific template arguments.
  ///
  /// First-level collection marks each group of template arguments
  /// (group contains template arguments that can be referred at the same time);
  /// second-level contains collection of template arguments in current group.
  using TArgsT = std::vector<std::vector<std::shared_ptr<DmngRsltTArg>>>;

  /// Type of collection with all scopes for resolution of template arguments.
  ///
  /// Parameters are resolved in later phase to specific template arguments.
  ///
  /// Each entry contains triple - boolean value and pair of indices:
  /// First boolean value indicates that the template argument scope requires
  /// late-referencing of template arguments.
  /// The second value: index that identifies parent scope,
  /// or -1 (converted to unsigned size type) if the scope has no parent scope;
  /// The third value: index that identifies argument group referred in selected
  /// scope, or -1 (converted to unsigned size type) if scope does not have
  /// argument group (yet).
  using TArgScopesT = std::vector<ParseState::TArgScopeT>;

  /// Type of variant temporary variable used by parser.
  using VariableT = ParseVariant<
    bool,
    int,
    unsigned int,
    long,
    unsigned long,
    long long,
    unsigned long long,

    DmngCvrQuals,
    DmngRefQuals,
    DmngTypeNameKind,

    std::string,
    std::shared_ptr<DmngRsltNode>,

    DmngRsltAdjustOffset,
    DmngRsltVendorQual,
    DmngRsltTArg
    >;

  /// Type of collection with variant temporary variables.
  using VariablesT = std::vector<VariableT>;


  /// \brief Report failure and switches to parent state.
  ///
  /// Invalidates current state. After this function current state may
  /// refer to invalid/non-existing state (dangling reference). Please use with
  /// caution.
  ///
  /// This function is intended to be used as state transition function. After
  /// this function control flow of parsing should move to end of iteration of
  /// parsing loop.
  void reportFailToParent() {
    assert(!States.empty() &&
           "This helper function is intended to be used only in parse loop.");
    if (!States.empty()) {
      States.top().resetContext();
      States.pop();
      if (!States.empty())
        States.top().fail();
    }
  }

  /// \brief Reports success to parent state, moves parse position to parse
  ///        position of current state and switches to parent state (in next
  ///        iteration).
  ///
  /// Invalidates current state. After this function current state may
  /// refer to invalid/non-existing state (dangling reference). Please use with
  /// caution.
  ///
  /// This function is intended to be used as state transition function. After
  /// this function control flow of parsing should move to end of iteration of
  /// parsing loop.
  void reportSuccessToParent() {
    assert(!States.empty() &&
           "This helper function is intended to be used only in parse loop.");
    if (!States.empty()) {
      auto NewPos = States.top().Pos;
      States.pop();
      if (!States.empty())
        States.top().advance(NewPos);
    }
  }

  /// \brief Reports success to parent state, moves parse position to specific
  ///        value and switches to parent state (in next iteration).
  ///
  /// Invalidates current state. After this function current state may
  /// refer to invalid/non-existing state (dangling reference). Please use with
  /// caution.
  ///
  /// This function is intended to be used as state transition function. After
  /// this function control flow of parsing should move to end of iteration of
  /// parsing loop.
  void reportSuccessToParent(ParsePosT NewPos) {
    assert(!States.empty() &&
           "This helper function is intended to be used only in parse loop.");
    if (!States.empty()) {
      States.pop();
      if (!States.empty())
        States.top().advance(NewPos);
    }
  }

  /// \brief Parses child state.
  ///
  /// Adds new child state. If child state failed, in next parsing loop
  /// iteration, it will detect failure and report it to parent.
  ///
  ///
  /// Invalidates current state. After this function current state may
  /// refer to invalid/non-existing state (dangling reference). Please use with
  /// caution.
  ///
  /// This function is intended to be used as state transition function. After
  /// this function control flow of parsing should move to end of iteration of
  /// parsing loop.
  void parseChildState(ParseStateId StateToParse) {
    if (States.empty()) {
      // ReSharper disable once CppUnreachableCode
      assert(false && "There is no current state.");
      return;
    }
    if (!States.top().DotSuccess) {
      reportFailToParent();
      return;
    }

    States.emplace(*this, StateToParse, States.top().Pos,
                   States.top().TArgCurGrpIdx, States.top().TArgScopeIdx,
                   States.top().GatherSubsEnabled);
  }

  /// \brief Parses child state.
  ///
  /// Adds new child state. If child state failed, in next parsing loop
  /// iteration, it will detect failure and move current state to next
  /// production.
  ///
  /// Invalidates current state. After this function current state may
  /// refer to invalid/non-existing state (dangling reference). Please use with
  /// caution.
  ///
  /// This function is intended to be used as state transition function. After
  /// this function control flow of parsing should move to end of iteration of
  /// parsing loop.
  void tryParseChildState(ParseStateId StateToParse,
                          unsigned NextProdDotPos,
                          bool ResetPosToStartPos = true) {
    if (States.empty()) {
      // ReSharper disable once CppUnreachableCode
      assert(false && "There is no current state.");
      return;
    }
    if (!States.top().DotSuccess) {
      States.top().moveToProduction(NextProdDotPos, ResetPosToStartPos);
      return;
    }

    States.emplace(*this, StateToParse, States.top().Pos,
                   States.top().TArgCurGrpIdx, States.top().TArgScopeIdx,
                   States.top().GatherSubsEnabled);
  }


  /// \brief Gets number of variables created for current (and child states).
  std::size_t getStateVarsCount() const {
    std::size_t Off = States.empty() ? 0 : States.top().VarsSize;

    return Variables.size() > Off ? Variables.size() - Off : 0;
  }

  /// \brief Gets variable of current state.
  const VariableT &getStateVar(std::size_t Idx) const {
    std::size_t Off = States.empty() ? 0 : States.top().VarsSize;

    assert(Idx + Off < Variables.size() && "Accessing variable out of bounds.");
    return Variables[Idx + Off];
  }

  /// \brief Gets variable of current state.
  VariableT &getStateVar(std::size_t Idx) {
    std::size_t Off = States.empty() ? 0 : States.top().VarsSize;

    assert(Idx + Off < Variables.size() && "Accessing variable out of bounds.");
    return Variables[Idx + Off];
  }

  /// \brief Tries to remove all variables added after current state
  ///        was created.
  void resetStateVars() {
    std::size_t Off = States.empty() ? 0 : States.top().VarsSize;

    if (Variables.size() > Off)
    Variables.resize(Off);
  }


  /// Constructs new instance of parse context with set mangled name in result.
  explicit ParseContext(const std::string &MangledName)
    : Result(MangledName) {}

  /// Constructs new instance of parse context with set mangled name in result.
  explicit ParseContext(std::string &&MangledName)
    : Result(std::move(MangledName)) {}

  /// Connects template parameters with template arguments referred by them.
  bool referenceTParams() {
    bool Success = true;

    const auto GroupsCount = TArgs.size();
    const auto ScopesCount = TArgScopes.size();

    // Resolve group-scope connections by tracking parent scope groups if
    // necessary.
    for (auto &&Scope : TArgScopes) {
      // The group is valid - no necessary to track ancestors.
      if (std::get<2>(Scope) < GroupsCount)
        continue;

      // Track ancestors scopes until scope with valid group is found.
      const auto *TrackedScope = &Scope;
      while (std::get<1>(*TrackedScope) < ScopesCount) {
        TrackedScope = &TArgScopes[std::get<1>(*TrackedScope)];

        if (std::get<2>(*TrackedScope) < GroupsCount) {
          std::get<2>(Scope) = std::get<2>(*TrackedScope);
          break;
        }
      }
    }

    // Resolve template parameters.
    for (auto &&TParam : TParams) {
      auto LateReferencing = std::get<0>(TParam);
      std::size_t GroupIdx;
      if (LateReferencing) {
        auto ScopeIdx = std::get<1>(TParam);
        if (ScopeIdx >= ScopesCount) {
          Success = false;
          // ReSharper disable once CppUnreachableCode
          assert(false && "Invalid template parameter (scope invalid).");
          continue;
        }

        GroupIdx = std::get<2>(TArgScopes[ScopeIdx]);
      }
      else {
        // Finish resolving early-referenced parameters.
        GroupIdx = std::get<1>(TParam);
      }

      if (GroupIdx >= GroupsCount) {
        Success = false;
        // ReSharper disable once CppUnreachableCode
        assert(false && "Invalid template parameter (group invalid).");
        continue;
      }

      const auto &TArgGroup = TArgs[GroupIdx];
      auto ArgIdx = std::get<2>(TParam)->getReferredTArgIdx();
      if (ArgIdx >= TArgGroup.size()) {
        Success = false;
        // ReSharper disable once CppUnreachableCode
        assert(false && "Invalid template parameter (arg invalid).");
        continue;
      }

      std::get<2>(TParam)->setReferredTArg(TArgGroup[ArgIdx]);
    }

    return Success;
  }

  /// Parse result.
  DmngRslt Result;

  /// Parse states.
  StatesT States;
  /// Mangling substitutions.
  SubsT Substitutions;
  /// Substitution identifiers (to provide uniqueness).
  SubsStrsT SubstitutionIds;
  /// Template parameters.
  TParamsT TParams;
  /// Template arguments (grouped values for template parameters).
  TArgsT TArgs;
  /// Scopes for template arguments referencing.
  TArgScopesT TArgScopes;
  /// Parse temporary variables.
  VariablesT Variables;
};

}

// -----------------------------------------------------------------------------
// Implementation

inline void ParseState::resetContext() {
  while (Ctx.Substitutions.size() > SubsSize) {
    Ctx.SubstitutionIds.erase(Ctx.Substitutions.back().first);
    Ctx.Substitutions.pop_back();
  }

  if (Ctx.TParams.size() > TParamsSize)
    Ctx.TParams.resize(TParamsSize);
  if (Ctx.TArgs.size() > TArgsSize)
    Ctx.TArgs.resize(TArgsSize);
  if (TArgCurGrpStartIdx < Ctx.TArgs.size() &&
      Ctx.TArgs[TArgCurGrpStartIdx].size() > TArgCurGrpSize)
    Ctx.TArgs[TArgCurGrpStartIdx].resize(TArgCurGrpSize);
  if (Ctx.TArgScopes.size() > TArgScopesSize)
    Ctx.TArgScopes.resize(TArgScopesSize);
  if (TArgScopeStartIdx < Ctx.TArgScopes.size())
    Ctx.TArgScopes[TArgScopeStartIdx] = TArgScopeStartVal;
  if (Ctx.Variables.size() > VarsSize)
    Ctx.Variables.resize(VarsSize);

  TArgCurGrpIdx = TArgCurGrpStartIdx;
  TArgScopeIdx = TArgScopeStartIdx;
}

inline void ParseState::addSubstitute(
    const std::shared_ptr<const DmngRsltNode> &SubRepr) {
  if (GatherSubsEnabled) {
    std::string SubId = Ctx.Result.getMangledName().substr(StartPos,
                                                           Pos - StartPos);

    if (Ctx.SubstitutionIds.insert(SubId).second) {
      Ctx.Substitutions.emplace_back(std::move(SubId), SubRepr->clone());
    }
  }
}

inline void ParseState::addSubstituteNoClone(
  std::shared_ptr<DmngRsltNode> &&SubRepr) {
  if (GatherSubsEnabled) {
    std::string SubId = Ctx.Result.getMangledName().substr(StartPos,
                                                           Pos - StartPos);

    if (Ctx.SubstitutionIds.insert(SubId).second) {
      Ctx.Substitutions.emplace_back(std::move(SubId), std::move(SubRepr));
    }
  }
}

inline void ParseState::addTArgGroup() {
  Ctx.TArgs.emplace_back();
  TArgCurGrpIdx = Ctx.TArgs.size() - 1;
}

inline void ParseState::addTArg(const std::shared_ptr<DmngRsltTArg> &Arg) {
  if (TArgCurGrpIdx >= Ctx.TArgs.size()) {
    // ReSharper disable once CppUnreachableCode
    assert(false && "There is no group to insert template argument into.");
    return;
  }

  Ctx.TArgs[TArgCurGrpIdx].push_back(Arg);
}

inline void ParseState::addTArg(std::shared_ptr<DmngRsltTArg> &&Arg) {
  if (TArgCurGrpIdx >= Ctx.TArgs.size()) {
    // ReSharper disable once CppUnreachableCode
    assert(false && "There is no group to insert template argument into.");
    return;
  }

  Ctx.TArgs[TArgCurGrpIdx].push_back(std::move(Arg));
}

inline void ParseState::addTArgScope(bool LateReferencing) {
  Ctx.TArgScopes.push_back(std::make_tuple(LateReferencing,
                                           TArgScopeIdx,
                                           static_cast<std::size_t>(-1)));
  TArgScopeIdx = Ctx.TArgScopes.size() - 1;
}

inline void ParseState::setTArgScope(bool LateReferencing) {
  if (TArgScopeIdx >= Ctx.TArgScopes.size()) {
    // ReSharper disable once CppUnreachableCode
    assert(false && "Current scope for template parameters is invalid.");
    return;
  }

  std::get<0>(Ctx.TArgScopes[TArgScopeIdx]) = LateReferencing;
}

inline void ParseState::assignTArgGroupToScope() {
  if (TArgCurGrpIdx >= Ctx.TArgs.size() ||
      TArgScopeIdx >= Ctx.TArgScopes.size()) {
    // ReSharper disable once CppUnreachableCode
    assert(false && "Template argument scope or group is invalid.");
    return;
  }

  std::get<2>(Ctx.TArgScopes[TArgScopeIdx]) = TArgCurGrpIdx;
}

std::pair<bool, std::size_t> ParseState::getTParamReferencingContext() const {
  const auto GroupsCount = Ctx.TArgs.size();
  const auto ScopesCount = Ctx.TArgScopes.size();

  if (TArgScopeIdx >= ScopesCount) {
    assert(false && "Template argument scope is invalid / undefined.");
    return std::make_pair(true, static_cast<std::size_t>(-1));
  }

  // Resolve group-scope connections by tracking parent scope groups if
  // necessary (early-referencing).
  auto TrackedScopeIdx = TArgScopeIdx;
  do {
    const auto &TrackedScope = Ctx.TArgScopes[TrackedScopeIdx];

    if (std::get<0>(TrackedScope))               // late-referencing case
      return std::make_pair(true, TrackedScopeIdx);
    if (std::get<2>(TrackedScope) < GroupsCount) // already referenced case
      return std::make_pair(false, std::get<2>(TrackedScope));

    TrackedScopeIdx = std::get<1>(TrackedScope); // parent scope tracking
  } while(TrackedScopeIdx < ScopesCount);

  // Fall-back to late-referencing if current attempt failed.
  return std::make_pair(true, TArgScopeIdx);
}

inline void ParseState::addTParam(
    const std::shared_ptr<DmngRsltTParamExpr> &Param) {
  auto RefCtx = getTParamReferencingContext();

  Ctx.TParams.push_back(std::make_tuple(RefCtx.first, RefCtx.second, Param));
}

inline void ParseState::addTParam(std::shared_ptr<DmngRsltTParamExpr> &&Param) {
  auto RefCtx = getTParamReferencingContext();

  Ctx.TParams.push_back(std::make_tuple(RefCtx.first, RefCtx.second,
                                        std::move(Param)));
}

inline ParseState::ParseState(ParseContext& Ctx, ParseStateId Id, ParsePosT Pos,
                              std::size_t TArgGrpIdx, std::size_t TArgScopeIdx,
                              bool GatherSubsEnabled)
  : Ctx(Ctx), StartPos(Pos), Pos(Pos), Id(Id), DotPos(0), DotSuccess(true),
    GatherSubsEnabled(GatherSubsEnabled), SubsSize(Ctx.Substitutions.size()),
    TParamsSize(Ctx.TParams.size()), TArgsSize(Ctx.TArgs.size()),
    TArgCurGrpSize(TArgGrpIdx < Ctx.TArgs.size() ? Ctx.TArgs[TArgGrpIdx].size()
                                                 : 0),
    TArgScopesSize(Ctx.TArgScopes.size()),
    TArgScopeStartVal(TArgScopeIdx < Ctx.TArgScopes.size()
                        ? Ctx.TArgScopes[TArgScopeIdx]
                        : TArgScopeT()),
    VarsSize(Ctx.Variables.size()),
    TArgCurGrpStartIdx(TArgGrpIdx), TArgCurGrpIdx(TArgGrpIdx),
    TArgScopeStartIdx(TArgScopeIdx), TArgScopeIdx(TArgScopeIdx) {}


// -----------------------------------------------------------------------------
// PARSE ROUTINE
// -----------------------------------------------------------------------------

// <<start>> ::= <mangled-name>
//
// <mangled-name> ::= _Z <encoding>
//
// <encoding> ::= <name> <bare-function-type>   # function name
//            ::= <name>                        # data name
//            ::= <special-name>
//
// <name> ::= <nested-name>
//        ::= <unscoped-name>
//        ::= <unscoped-template-name> <template-args>
//        ::= <local-name>
//
// <bare-function-type> ::= <type> <bare-function-type>   # signature types
//                      ::= <type>
//
// <special-name> ::= TV <type>   # virtual table
//                ::= TT <type>   # VTT structure (construction vtable index)
//                ::= TI <type>   # typeinfo structure
//                ::= TS <type>   # typeinfo name (null-terminated byte string)
//                ::= T <call-offset> <encoding>   # base is the nominal target
//                                                 # function of thunk
//                ::= Tc <call-offset> <call-offset> <encoding>
//                       # base is the nominal target function of thunk
//                       # first call-offset is 'this' adjustment
//                       # second call-offset is result adjustment
//                ::= GV <name>   # Guard variable for one-time
//                                # initialization (no <type>, only object name)
//                ::= GR <name> _             # First LE temporary
//                       # name is object name initialized by temporary
//                ::= GR <name> <seq-id> _    # Subsequent LE temporaries
//
//
// <nested-name> ::= N [<vendor-qualifiers>] [<CV-qualifiers>]
//                   [<vendor-qualifiers>] [<ref-qualifier>]
//                   [<vendor-qualifiers>] <prefix> <unqualified-name> E
//               ::= N [<vendor-qualifiers>] [<CV-qualifiers>]
//                   [<vendor-qualifiers>] [<ref-qualifier>]
//                   [<vendor-qualifiers>] <template-prefix> <template-args> E
//
// <unscoped-name> ::= <unqualified-name>
//                 ::= St <unqualified-name>   # ::std::
//
// <unscoped-template-name> ::= <unscoped-name>
//                          ::= <substitution>
//
// <template-args> ::= I <template-arg>+ E
//
// <local-name> ::= Z <encoding> E <name> [<discriminator>]
//              ::= Z <encoding> E s [<discriminator>]   # string literal
//              ::= Z <encoding> Ed [ <number> ] _ <name>
//                     # <encoding> contains function name, <name> describes
//                     # local entity, <number> is non-negative index of
//                     # parameter in reverse order (last has no index)
//
// <type> ::= <builtin-type>
//        ::= <function-type>
//        ::= <class-enum-type>
//        ::= <array-type>
//        ::= <vector-type>
//        ::= <pointer-to-member-type>
//        ::= <template-param>
//        ::= <template-template-param> <template-args>
//        ::= <decltype>
//        ::= <substitution>
//        ::= <CV-qualifiers> <type> # substitute as group with vendor-extended
//                                   # qualifiers, <CV-qualifiers> is not empty.
//        ::= P <type>   # pointer-to
//        ::= R <type>   # reference-to
//        ::= O <type>   # rvalue reference-to (C++0x)
//        ::= C <type>   # complex pair (C 2000)
//        ::= G <type>   # imaginary (C 2000)
//        ::= <vendor-qualifiers> <type>
//               # vendor-extended type qualifier (set is substituted as group
//               # with cvr-qualifiers).
//        ::= Dp <type>   # pack expansion (C++0x)
//
// <call-offset> ::= h <nv-offset> _
//               ::= v <v-offset> _
//
// <nv-offset> ::= <number>   # non-virtual base override
//
// <v-offset>  ::= <number> _ <number>
//                   # virtual base override, with vcall offset
//
// <vendor-qualifiers> ::= <vendor-qualifier>+
//
// <prefix> ::= <unqualified-name>                  # global class or namespace
//          ::= <prefix> <unqualified-name>         # nested class or namespace
//          ::= <template-prefix> <template-args>   # class template
//                                                  # specialization
//          ::= <template-param>                    # template type parameter
//          ::= <decltype>                          # decltype qualifier
//          ::= <prefix> <data-member-prefix>       # initializer of a data
//                                                  # member
//          ::= <substitution>
//
// <unqualified-name> ::= <operator-name>
//                    ::= <ctor-dtor-name>
//                    ::= <source-name>
//                    ::= <unnamed-type-name>
//
// <template-prefix> ::= <unqualified-name>            # global template
//                   ::= <prefix> <unqualified-name>   # nested template
//                   ::= <template-param>   # template template parameter
//                   ::= <substitution>
//
//
// <substitution> ::= S <seq-id> _
//                ::= S_
//                ::= St   # ::std::
//                ::= Sa   # ::std::allocator
//                ::= Sb   # ::std::basic_string
//                ::= Ss   # ::std::basic_string <char,
//                         #   ::std::char_traits<char>, ::std::allocator<char>>
//                ::= Si   # ::std::basic_istream<char, std::char_traits<char>>
//                ::= So   # ::std::basic_ostream<char, std::char_traits<char>>
//                ::= Sd   # ::std::basic_iostream<char, std::char_traits<char>>
//
// <template-arg> ::= <type>                # type or template
//                ::= X <expression> E      # expression
//                ::= <expr-primary>        # simple expressions
//                ::= J <template-arg>* E   # argument pack
//
// <discriminator> ::= _ <number>      # when non-negative number < 10
//                 ::= __ <number> _   # when non-negative number >= 10
//
// <builtin-type> ::= <fixed-builtin-type>
//                ::= u <source-name>       # vendor-extended type
//
// <function-type> ::= [<vendor-qualifiers>] [<CV-qualifiers>]
//                     [<vendor-qualifiers>] F [Y] <bare-function-type>
//                     [<ref-qualifier>] E
//
// <class-enum-type> ::= <name>     # non-dependent type name, dependent type
//                                  #   name, or dependent typename-specifier
//                   ::= Ts <name>  # dependent elaborated type specifier
//                                  #   using 'struct' or 'class'
//                   ::= Tu <name>  # dependent elaborated type specifier
//                                  #   using 'union'
//                   ::= Te <name>  # dependent elaborated type specifier
//                                  #   using 'enum'
//
// <array-type> ::= A <number> _ <type>
//              ::= A [<expression>] _ <type>
//                    # array of element types with dimension expression or
//                    # positive dimension number
//
// <vector-type> ::= Dv <number> _ <vec-ext-type>
//               ::= Dv [<expression>] _ <vec-ext-type>
//                     # vector of element types with dimension expression or
//                     # positive dimension number
//
// <vec-ext-type> ::= <type>
//                ::= p       # AltiVec pixel
//
// <pointer-to-member-type> ::= M <type> <type>   # class and member types
//
// <template-param> ::= T_             # first template parameter
//                  ::= T <number> _   # parameter-2 non-negative number
//
// <template-template-param> ::= <template-param>
//                           ::= <substitution>
//
// <decltype> ::= Dt <expression> E   # decltype of an id-expression
//                                    # or class member access (C++0x)
//            ::= DT <expression> E   # decltype of an expression (C++0x)
//
// <vendor-qualifier> ::= U <source-name> [<template-args>]
//
// <data-member-prefix> ::= <source-name> M # name of member
//
// <unnamed-type-name> ::= Ut [<number>] _  # non-negative number
//                     ::= <closure-type-name>
//
// <closure-type-name> ::= Ul <lambda-sig> E [<number>] _ # non-negative number
//
// <lambda-sig> ::= <type>+
//                    # Parameter types or "v" if the lambda has no parameters
//
// <operator-name> ::= <fixed-operator-name>
//                 ::= cv <type>                  # (cast)
//                 ::= li <source-name>           # operator ""
//                 ::= v <digit> <source-name>    # vendor extended operator
//
// Limited expression support:
//  - only primary non-floating point.
//
// <expression> ::= <expr-primary>   # only primary expressions and
//              ::= <template-param> # template parameters are supported
//
// <expr-primary> ::= L <type> <number> E   # integer literal
//                ::= L <type> E            # string literal
//                ::= L <type> E            # nullptr literal (i.e., "LDnE")
//                ::= L <type> 0 E          # null pointer template argument
//                ::= L <mangled-name> E    # external name

/// \brief Parses mangled name (by move).
DmngRslt ItaniumNameParser::parse(std::string &&MangledName) {
  using ParseNodeT = std::shared_ptr<DmngRsltNode>;

  ParseContext Ctx(std::move(MangledName));


  // Standard substitutions.
  std::unordered_map<std::string, std::shared_ptr<const DmngRsltNode>>
    StdSubstitutions;
  {
    // [NamePart] ::std
    auto StNdPart = std::make_shared<DmngRsltSrcNamePart>("std");

    // [NameParts] ::std::allocator
    auto SaNdParts = std::make_shared<DmngRsltNameParts>(StNdPart);
    SaNdParts->addPart(std::make_shared<DmngRsltSrcNamePart>("allocator"));

    // [NameParts] ::std::basic_string
    auto SbNdParts = std::make_shared<DmngRsltNameParts>(StNdPart);
    SbNdParts->addPart(std::make_shared<DmngRsltSrcNamePart>("basic_string"));

    // [Type] char
    auto CharNdType = std::make_shared<DmngRsltBuiltinType>(DBT_Char);

    // [NameParts, Type] ::std::char_traits<char>
    auto CTraitsCNdParts = std::make_shared<DmngRsltNameParts>(StNdPart);
    auto CTraitsCNdLPart = std::make_shared<DmngRsltSrcNamePart>("char_traits");
    CTraitsCNdLPart->addTemplateArg(DmngRsltTArg(CharNdType));
    CTraitsCNdParts->addPart(std::move(CTraitsCNdLPart));
    auto CTraitsCNdType = adaptNode<DmngRsltType>(CTraitsCNdParts);

    // [NameParts, Type] ::std::allocator<char>
    auto AllocCNdParts = SaNdParts->clone(true)->getAs<DNDK_NameParts>();
    AllocCNdParts->getModifiableLastPart()
      ->addTemplateArg(DmngRsltTArg(CharNdType));
    auto AllocCNdType = adaptNode<DmngRsltType>(AllocCNdParts);

    // [NameParts] ::std::basic_string<char,
    //                                 ::std::char_traits<char>,
    //                                 ::std::allocator<char>>
    auto SsNdParts = SbNdParts->clone(true)->getAs<DNDK_NameParts>();
    auto &&SsNdLPart = SsNdParts->getModifiableLastPart();
    SsNdLPart->addTemplateArg(DmngRsltTArg(CharNdType));
    SsNdLPart->addTemplateArg(DmngRsltTArg(CTraitsCNdType));
    SsNdLPart->addTemplateArg(DmngRsltTArg(AllocCNdType));

    // [NameParts] ::std::basic_istream<char, ::std::char_traits<char>>
    auto SiNdParts = std::make_shared<DmngRsltNameParts>(StNdPart);
    auto SiNdLPart = std::make_shared<DmngRsltSrcNamePart>("basic_istream");
    SiNdLPart->addTemplateArg(DmngRsltTArg(CharNdType));
    SiNdLPart->addTemplateArg(DmngRsltTArg(CTraitsCNdType));
    SiNdParts->addPart(std::move(SiNdLPart));

    // [NameParts] ::std::basic_ostream<char, ::std::char_traits<char>>
    auto SoNdParts = std::make_shared<DmngRsltNameParts>(StNdPart);
    auto SoNdLPart = std::make_shared<DmngRsltSrcNamePart>("basic_ostream");
    SoNdLPart->addTemplateArg(DmngRsltTArg(CharNdType));
    SoNdLPart->addTemplateArg(DmngRsltTArg(CTraitsCNdType));
    SoNdParts->addPart(std::move(SoNdLPart));

    // [NameParts] ::std::basic_iostream<char, ::std::char_traits<char>>
    auto SdNdParts = std::make_shared<DmngRsltNameParts>(StNdPart);
    auto SdNdLPart = std::make_shared<DmngRsltSrcNamePart>("basic_iostream");
    SdNdLPart->addTemplateArg(DmngRsltTArg(CharNdType));
    SdNdLPart->addTemplateArg(DmngRsltTArg(CTraitsCNdType));
    SdNdParts->addPart(std::move(SdNdLPart));


    // <std-sub> ::= St   # ::std
    //           ::= Sa   # ::std::allocator
    //           ::= Sb   # ::std::basic_string
    //           ::= Ss   # ::std::basic_string <char,
    //                    #   ::std::char_traits<char>, ::std::allocator<char>>
    //           ::= Si   # ::std::basic_istream<char, std::char_traits<char>>
    //           ::= So   # ::std::basic_ostream<char, std::char_traits<char>>
    //           ::= Sd   # ::std::basic_iostream<char, std::char_traits<char>>
    StdSubstitutions["St"] = StNdPart;
    StdSubstitutions["Sa"] = SaNdParts;
    StdSubstitutions["Sb"] = SbNdParts;
    StdSubstitutions["Ss"] = SsNdParts;
    StdSubstitutions["Si"] = SiNdParts;
    StdSubstitutions["So"] = SoNdParts;
    StdSubstitutions["Sd"] = SdNdParts;
  }

  #define INP_ENSURE_TMP_SIZE(Size_)                                           \
  do {                                                                         \
    if (Ctx.getStateVarsCount() < (Size_)) {                                   \
      assert(false && "Not enough temporary variables to generate result.");   \
      return Ctx.Result.setFailed();                                           \
    }                                                                          \
  } while(false)


  Ctx.States.emplace(Ctx, PSI_Start);

  int CC = 0;

  while (Ctx.States.size() > 0 && Ctx.Result.isSuccessful()) {
    ++CC;
    auto& State = Ctx.States.top(); // Current state.

    switch (State.Id) {

    // <<start>> ::= <mangled-name>
    case PSI_Start:
      if (!State.DotSuccess)
        return Ctx.Result.setFailed();

      switch (State.DotPos) {
      // <<start>> ::= @ <mangled-name>
      case 0:
        Ctx.parseChildState(PSI_MangledName);
        break;
      // <<start>> ::= <mangled-name> @
      case 1:
        if (State.Pos == Ctx.Result.getMangledName().size()) {
          INP_ENSURE_TMP_SIZE(1);

          const auto& NdMName =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();
          Ctx.Result.setName(NdMName->getAs<DNDK_Name>());

          Ctx.resetStateVars();

          Ctx.reportSuccessToParent(); // Ending parsing.
          break;
        }
        return Ctx.Result.setFailed();

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <mangled-name> ::= _Z <encoding>
    //
    // Result variables:   DmngRsltName
    case PSI_MangledName:
      switch (State.DotPos) {
      // <mangled-name> ::= @ _Z <encoding>
      case 0:
        State.addTArgScope();
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "_Z"))
          State.advance();
        else
          Ctx.reportFailToParent();
        break;
      // <mangled-name> ::= _Z @ <encoding>
      case 1:
        Ctx.parseChildState(PSI_Encoding);
        break;
      // <mangled-name> ::= _Z <encoding> @
      case 2:
        Ctx.reportSuccessToParent();
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;

    // MC***
    // <mangled-name--bug> ::= ( _Z | Z ) <encoding>
    //
    // Result variables:   DmngRsltName
    case PSI_MangledName_Bug:
      switch (State.DotPos) {
      // <mangled-name--bug> ::= @ ( _Z | Z ) <encoding>
      case 0:
        State.addTArgScope();
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "_Z"))
          State.advance();
        else if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "Z"))
          State.advance();
        else
          Ctx.reportFailToParent();
        break;
      // <mangled-name--bug> ::= ( _Z | Z ) @ <encoding>
      case 1:
        Ctx.parseChildState(PSI_Encoding);
        break;
      // <mangled-name--bug> ::= ( _Z | Z ) <encoding> @
      case 2:
        Ctx.reportSuccessToParent();
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <encoding> ::= <name> <bare-function-type> # function name
    //            ::= <name>                      # data name
    //            ::= <special-name>
    //
    // Result variables:   DmngRsltName
    case PSI_Encoding:
      switch (State.DotPos) {
      // <encoding> ::= @ <name> <bare-function-type>
      //            ::= @ <name>
      case 0:
        Ctx.tryParseChildState(PSI_Name, 4);
        break;
      // <encoding> ::= <name> @ <bare-function-type>
      case 1:
        Ctx.tryParseChildState(PSI_BareFunctionType, 3, false);
        break;
      // <encoding> ::= <name> <bare-function-type> @
      case 2: {
          INP_ENSURE_TMP_SIZE(2);

          auto NdOName = Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>()
            ->getAs<DNDK_Name>()->getAs<DNK_Ordinary>();

          for (std::size_t I = 1, E = Ctx.getStateVarsCount(); I < E; ++I) {
            const auto& NdType =
              Ctx.getStateVar(I).unsafeGetAsExact<ParseNodeT>();
            NdOName->addSignatureType(NdType->getAs<DNDK_Type>());
          }

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdOName));

          Ctx.reportSuccessToParent();
        }
        break;
      // <encoding> ::= <name> @
      case 3:
        // If result node has no signature, it is treated as data node.
        Ctx.reportSuccessToParent();
        break;

      // <encoding> ::= @ <special-name>
      case 4:
        Ctx.parseChildState(PSI_SpecialName);
        break;
      // <encoding> ::= <special-name> @
      case 5:
        Ctx.reportSuccessToParent();
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;

    // MC***
    // <name> ::= <nested-name>
    //        ::= <unscoped-template-name> <template-args> # before <uns-name>
    //        ::= <unscoped-name>
    //        ::= <local-name>
    //
    // Result variables:   DmngRsltName (DmngRsltOrdinaryName)
    case PSI_Name:
      switch (State.DotPos) {
      // <name> ::= @ <nested-name>
      case 0:
        Ctx.tryParseChildState(PSI_NestedName, 2);
        break;
      // <name> ::= <nested-name> @
      case 1:
        Ctx.reportSuccessToParent();
        break;

      // <name> ::= @ <unscoped-template-name> <template-args>
      case 2:
        Ctx.tryParseChildState(PSI_UnscopedTemplateName, 5);
        break;
      // <name> ::= <unscoped-template-name> @ <template-args>
      case 3:
        Ctx.tryParseChildState(PSI_TemplateArgs, 5);
        break;
      // <name> ::= <unscoped-template-name> <template-args> @
      case 4: {
          INP_ENSURE_TMP_SIZE(2);

          auto NdOName = Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>()
            ->getAs<DNDK_Name>()->getAs<DNK_Ordinary>();
          auto NdONameCopy = NdOName->clone(true);
          auto &&NdTNPart = NdONameCopy->getModifiableLastPart();

          for (std::size_t I = 1, E = Ctx.getStateVarsCount(); I < E; ++I) {
            auto TArg = Ctx.getStateVar(I).unsafeGetAsExact<DmngRsltTArg>();
            NdTNPart->addTemplateArg(std::move(TArg));
          }

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdONameCopy));

          Ctx.reportSuccessToParent();
        }
        break;

      // <name> ::= @ <unscoped-name>
      case 5:
        Ctx.tryParseChildState(PSI_UnscopedName, 7);
        break;
      // <name> ::= <unscoped-name> @
      case 6:
        Ctx.reportSuccessToParent();
        break;

      // <name> ::= @ <local-name>
      case 7:
        Ctx.parseChildState(PSI_LocalName);
        break;
      // <name> ::= <local-name> @
      case 8:
        Ctx.reportSuccessToParent();
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <bare-function-type> ::= <type>+
    //
    // Result variables:   DmngRsltType+
    case PSI_BareFunctionType:
      switch (State.DotPos) {
      // <bare-function-type> ::= @ <type>+
      case 0:
        Ctx.parseChildState(PSI_Type);
        break;
      // <bare-function-type> ::= <type>+ @ <type>*
      case 1:
        Ctx.tryParseChildState(PSI_Type, 3, false);
        break;
      case 2:
        // Flattening recursive production.
        State.reset(false);
        State.moveToProduction(1, false);
        break;
      // <bare-function-type> ::= <type>+ @
      case 3:
        Ctx.reportSuccessToParent();
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <special-name> ::= TV <type>
    //                ::= TT <type>
    //                ::= TI <type>
    //                ::= TS <type>
    //                ::= T <call-offset> <encoding>
    //                ::= Tc <call-offset> <call-offset> <encoding>
    //                ::= GV <name>
    //                ::= GR <name> [<seq-id>] _
    //
    // Result variables:   DmngRsltName (DmngRsltSpecialName)
    case PSI_SpecialName:
      switch (State.DotPos) {
      // <special-name> ::= @ TV <type>
      case 0:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "TV"))
          State.advance();
        else
          State.moveToProduction(3);
        break;
      // <special-name> ::= TV @ <type>
      case 1:
        Ctx.tryParseChildState(PSI_Type, 3);
        break;
      // <special-name> ::= TV <type> @
      case 2: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdType =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();

          auto NdSName = std::make_shared<DmngRsltSpecialName>(
            DmngRsltSpecialName::createVirtualTable(
              NdType->getAs<DNDK_Type>()));

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdSName));

          Ctx.reportSuccessToParent();
        }
        break;

      // <special-name> ::= @ TT <type>
      case 3:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "TT"))
          State.advance();
        else
          State.moveToProduction(6);
        break;
      // <special-name> ::= TT @ <type>
      case 4:
        Ctx.tryParseChildState(PSI_Type, 6);
        break;
      // <special-name> ::= TT <type> @
      case 5: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdType =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();

          auto NdSName = std::make_shared<DmngRsltSpecialName>(
            DmngRsltSpecialName::createVirtualTableTable(
              NdType->getAs<DNDK_Type>()));

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdSName));

          Ctx.reportSuccessToParent();
        }
        break;

      // <special-name> ::= @ TI <type>
      case 6:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "TI"))
          State.advance();
        else
          State.moveToProduction(9);
        break;
      // <special-name> ::= TI @ <type>
      case 7:
        Ctx.tryParseChildState(PSI_Type, 9);
        break;
      // <special-name> ::= TI <type> @
      case 8: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdType =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();

          auto NdSName = std::make_shared<DmngRsltSpecialName>(
            DmngRsltSpecialName::createTypeInfoStruct(
              NdType->getAs<DNDK_Type>()));

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdSName));

          Ctx.reportSuccessToParent();
        }
        break;

      // <special-name> ::= @ TS <type>
      case 9:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "TS"))
          State.advance();
        else
          State.moveToProduction(12);
        break;
      // <special-name> ::= TS @ <type>
      case 10:
        Ctx.tryParseChildState(PSI_Type, 12);
        break;
      // <special-name> ::= TS <type> @
      case 11: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdType =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();

          auto NdSName = std::make_shared<DmngRsltSpecialName>(
            DmngRsltSpecialName::createTypeInfoNameString(
              NdType->getAs<DNDK_Type>()));

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdSName));

          Ctx.reportSuccessToParent();
        }
        break;

      // <special-name> ::= @ T <call-offset> <encoding>
      case 12:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(), State.Pos, "T"))
          State.advance();
        else
          State.moveToProduction(16);
        break;
      // <special-name> ::= T @ <call-offset> <encoding>
      case 13:
        Ctx.tryParseChildState(PSI_CallOffset, 16);
        break;
      // <special-name> ::= T <call-offset> @ <encoding>
      case 14:
        Ctx.tryParseChildState(PSI_Encoding, 16);
        break;
      // <special-name> ::= T <call-offset> <encoding> @
      case 15: {
          INP_ENSURE_TMP_SIZE(2);

          const auto &ThisOff =
            Ctx.getStateVar(0).unsafeGetAsExact<DmngRsltAdjustOffset>();
          const auto &NdEnc =
            Ctx.getStateVar(1).unsafeGetAsExact<ParseNodeT>();

          auto NdSName = std::make_shared<DmngRsltSpecialName>(
            DmngRsltSpecialName::createVirtualThunk(
              NdEnc->getAs<DNDK_Name>(), ThisOff));

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdSName));

          Ctx.reportSuccessToParent();
        }
        break;

      // <special-name> ::= @ Tc <call-offset> <call-offset> <encoding>
      case 16:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "Tc"))
          State.advance();
        else
          State.moveToProduction(21);
        break;
      // <special-name> ::= Tc @ <call-offset> <call-offset> <encoding>
      case 17:
        Ctx.tryParseChildState(PSI_CallOffset, 21);
        break;
      // <special-name> ::= Tc <call-offset> @ <call-offset> <encoding>
      case 18:
        Ctx.tryParseChildState(PSI_CallOffset, 21);
        break;
      // <special-name> ::= Tc <call-offset> <call-offset> @ <encoding>
      case 19:
        Ctx.tryParseChildState(PSI_Encoding, 21);
        break;
      // <special-name> ::= Tc <call-offset> <call-offset> <encoding> @
      case 20: {
          INP_ENSURE_TMP_SIZE(3);

          const auto &ThisOff =
            Ctx.getStateVar(0).unsafeGetAsExact<DmngRsltAdjustOffset>();
          const auto &RetOff =
            Ctx.getStateVar(1).unsafeGetAsExact<DmngRsltAdjustOffset>();
          const auto &NdEnc =
            Ctx.getStateVar(2).unsafeGetAsExact<ParseNodeT>();

          auto NdSName = std::make_shared<DmngRsltSpecialName>(
            DmngRsltSpecialName::createVirtualThunk(
              NdEnc->getAs<DNDK_Name>(), ThisOff, RetOff));

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdSName));

          Ctx.reportSuccessToParent();
        }
        break;


      // <special-name> ::= @ GV <name>
      case 21:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "GV"))
          State.advance();
        else
          State.moveToProduction(24);
        break;
      // <special-name> ::= GV @ <name>
      case 22:
        Ctx.tryParseChildState(PSI_Name, 24);
        break;
      // <special-name> ::= GV <name> @
      case 23: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdName =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();

          auto NdSName = std::make_shared<DmngRsltSpecialName>(
            DmngRsltSpecialName::createGuardVariable(
              NdName->getAs<DNDK_Name>()->getAs<DNK_Ordinary>()));

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdSName));

          Ctx.reportSuccessToParent();
        }
        break;

      // <special-name> ::= @ GR <name> [<seq-id>] _
      case 24:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "GR"))
          State.advance();
        else
          Ctx.reportFailToParent();
        break;
      // <special-name> ::= GR @ <name> [<seq-id>] _
      case 25:
        Ctx.parseChildState(PSI_Name);
        break;
      // <special-name> ::= GR <name> @ [<seq-id>] _
      case 26: {
          unsigned long long TemporaryId = 0;
          if (matchSeqIdAndAdvance(Ctx.Result.getMangledName(),
                                    State.Pos, TemporaryId)) {
            assert(TemporaryId + 1 > TemporaryId && "Id out of bounds.");
            ++TemporaryId;
          }
          Ctx.Variables.emplace_back(TemporaryId);
          State.advance();
        }
        break;
      // <special-name> ::= GR <name> [<seq-id>] @ _
      case 27:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "_"))
          State.advance();
        else
          Ctx.reportFailToParent();
        break;
      // <special-name> ::= GR <name> [<seq-id>] _ @
      case 28: {
          INP_ENSURE_TMP_SIZE(2);

          const auto &NdName =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();
          auto Id =
            Ctx.getStateVar(1).unsafeGetAsExact<unsigned long long>();

          auto NdSName = std::make_shared<DmngRsltSpecialName>(
            DmngRsltSpecialName::createLifeExtTemporary(
              NdName->getAs<DNDK_Name>()->getAs<DNK_Ordinary>(), Id));

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdSName));

          Ctx.reportSuccessToParent();
        }
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <nested-name> ::= N [<vendor-qualifiers>] [<CV-qualifiers>]
    //                   [<vendor-qualifiers>] [<ref-qualifier>]
    //                   [<vendor-qualifiers>] <nested-name--sfx>
    //
    // Nested name suffix reflows: <prefix>, <temple-prefix>, <template-args>,
    //                             <unqualified-name>
    // in <nested-name> in more relaxed way - it requires final semantic
    // check:
    // - name parts: at least 2 or 1 if last part is template
    // - last part is not data member name.
    //
    // <nested-name--start> ::= <unqualified-name>
    //                      ::= <template-param>
    //                      ::= <decltype>
    //
    // <nested-name--step> ::= <data-member-prefix>
    //                     ::= <unqualified-name>
    //
    // <nested-name--sfx> ::= ( <nested-name--start> [() <template-args>] |
    //                          <substitution> [<template-args>] )
    //                        (() <nested-name--step> [() <template-args>])* E
    // () - position to register prefix as substitution.
    //
    // Result variables:   DmngRsltOrdinaryName
    case PSI_NestedName:
      switch (State.DotPos) {
      // <nested-name> ::= @ N [<vendor-qualifiers>] [<CV-qualifiers>] ...
      case 0:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "N"))
          State.advance();
        else
          Ctx.reportFailToParent();
        break;
      // <nested-name> ::= N @ [<vendor-qualifiers>] [<CV-qualifiers>] ...
      case 1:
        // Phase 1 of collecting all vendor-extended qualifiers.
        Ctx.tryParseChildState(PSI_VendorQuals, 2, false);
        break;
      // <nested-name> ::= N [<vendor-qualifiers>] @ [<CV-qualifiers>] ...
      case 2: {
          DmngCvrQuals CvrQuals;
          if (matchCvrQualsAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, CvrQuals)) {
            Ctx.Variables.emplace_back(CvrQuals);
          }
          State.advance();
        }
        break;
      // <nested-name> ::= N [<vendor-qualifiers>] [<CV-qualifiers>] @
      //                   [<vendor-qualifiers>] ...
      case 3:
        // Phase 2 of collecting all vendor-extended qualifiers.
        Ctx.tryParseChildState(PSI_VendorQuals, 4, false);
        break;
      // <nested-name> ::= ... [<vendor-qualifiers>] @ [<ref-qualifier>] ...
      case 4: {
          DmngRefQuals RefQuals;
          if (matchRefQualsAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, RefQuals)) {
            Ctx.Variables.emplace_back(RefQuals);
          }
          State.advance();
        }
        break;
      // <nested-name> ::= ... [<ref-qualifier>] @ [<vendor-qualifiers>] ...
      case 5:
        // Phase 3 of collecting all vendor-extended qualifiers.
        Ctx.tryParseChildState(PSI_VendorQuals, 6, false);
        break;
      // <nested-name> ::= ... [<vendor-qualifiers>] @ <nested-name--sfx>
      case 6:
        Ctx.parseChildState(PSI_NestedName_Sfx);
        break;
      // <nested-name> ::= ... [<vendor-qualifiers>] <nested-name--sfx> @
      case 7: {
          INP_ENSURE_TMP_SIZE(1);

          std::size_t QualsCount = Ctx.getStateVarsCount() - 1;

          auto &&NdNParts =
            Ctx.getStateVar(QualsCount).unsafeGetAsExact<ParseNodeT>()
              ->getAs<DNDK_NameParts>();

          // Check semantics of nested name.
          if (NdNParts->getParts().empty() ||
              NdNParts->getLastPart()->isDataMember() ||
              (!NdNParts->getLastPart()->isTemplate() &&
                NdNParts->getParts().size() < 2)) {
            Ctx.reportFailToParent();
            break;
          }

          auto NdOName =
            std::make_shared<DmngRsltOrdinaryName>(std::move(NdNParts));

          for (std::size_t I = 0; I < QualsCount; ++I) {
            const auto& Qual = Ctx.getStateVar(I);

            if (Qual.isAExact<DmngRsltVendorQual>()) {
              const auto &VQual = Qual.unsafeGetAsExact<DmngRsltVendorQual>();
              auto ASQual = ASExtractFunc(VQual);

              if (ASQual != DASQ_None)
                NdOName->setAsQuals(ASQual);
              NdOName->addVendorQual(VQual);
            }
            else if(Qual.isAExact<DmngCvrQuals>())
              NdOName->setCvrQuals(Qual.unsafeGetAsExact<DmngCvrQuals>());
            else if(Qual.isAExact<DmngRefQuals>())
              NdOName->setRefQuals(Qual.unsafeGetAsExact<DmngRefQuals>());
          }

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdOName));

          Ctx.reportSuccessToParent();
        }
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;

    // MC***
    // <nested-name--sfx> ::= ( <nested-name--start> [() <template-args>] |
    //                          <substitution> [<template-args>] )
    //                        (() <nested-name--step> [() <template-args>])* E
    //
    // Result variables:   DmngRsltNameParts
    case PSI_NestedName_Sfx:
      switch (State.DotPos) {
      // <nested-name--sfx> ::= ( @ <nested-name--start> [() <template-args>]
      //                     | <substitution> [<template-args>] ) ( ... )* E
      case 0:
        Ctx.tryParseChildState(PSI_NestedName_Start, 2);
        break;
      // <nested-name--sfx> ::= ( <nested-name--start> @ [() <template-args>]
      //                     | <substitution> [<template-args>] ) ( ... )* E
      case 1: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdNode =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();
          auto NdNParts = adaptNode<DmngRsltNameParts>(NdNode);

          if (NdNParts == nullptr) {
            Ctx.reportFailToParent();
            break;
          }

          // Simple check to omit last non-terminal in substitution set.
          if (!matchPrefix(Ctx.Result.getMangledName(), State.Pos, "E")) {
            // Add original non-adapted as substitute (to allow correct
            // template parameter/decltype() referencing).
            State.addSubstitute(NdNode);
          }

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdNParts));

          State.moveToProduction(4, false);
        }
        break;

      // <nested-name--sfx> ::= ( <nested-name--start> [() <template-args>]
      //                     | @ <substitution> [<template-args>] ) ( ... )* E
      case 2:
        Ctx.parseChildState(PSI_Substitution);
        break;
      // <nested-name--sfx> ::= ( <nested-name--start> [() <template-args>]
      //                     | <substitution> @ [<template-args>] ) ( ... )* E
      case 3: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdNode =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();
          auto NdNParts = adaptNode<DmngRsltNameParts>(NdNode);

          if (NdNParts == nullptr) {
            Ctx.reportFailToParent();
            break;
          }

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdNParts));

          State.moveToProduction(4, false);
        }
        break;

      // <nested-name--sfx> ::= ( <nested-name--start> [() @ <template-args>]
      //                     | <substitution> [@ <template-args>] ) ( ... )* E
      case 4:
        Ctx.tryParseChildState(PSI_TemplateArgs, 6, false);
        break;

      // <nested-name--sfx> ::= ( <nested-name--start> [() <template-args>]
      //                        | <substitution> [<template-args>] ) @
      //                        ( () <nested-name--step>
      //                          [() <template-args>] )* E
      case 5: {
          INP_ENSURE_TMP_SIZE(2);

          auto NdNParts = Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>()
            ->getAs<DNDK_NameParts>();
          auto NdNPartsCopy = NdNParts->clone(true)->getAs<DNDK_NameParts>();
          auto &&NdTNPart = NdNPartsCopy->getModifiableLastPart();

          for (std::size_t I = 1, E = Ctx.getStateVarsCount(); I < E; ++I) {
            auto TArg = Ctx.getStateVar(I).unsafeGetAsExact<DmngRsltTArg>();
            NdTNPart->addTemplateArg(std::move(TArg));
          }

          // Simple check to omit last non-terminal in substitution set.
          if (!matchPrefix(Ctx.Result.getMangledName(), State.Pos, "E")) {
            State.addSubstitute(NdNPartsCopy);
          }

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdNPartsCopy));

          State.advance();
        }
        break;

      // <nested-name--sfx> ::= ( ... ) ( () @ <nested-name--step>
      //                                  [() <template-args>] )* E
      case 6:
        Ctx.tryParseChildState(PSI_NestedName_Step, 11, false);
        break;
      // <nested-name--sfx> ::= ( ... ) ( () <nested-name--step> @
      //                                  [() <template-args>] )* E
      case 7: {
          INP_ENSURE_TMP_SIZE(2);

          auto NdNParts = Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>()
            ->getAs<DNDK_NameParts>();
          auto &&NdLPart = Ctx.getStateVar(1).unsafeGetAsExact<ParseNodeT>()
            ->getAs<DNDK_NamePart>();

          NdNParts->addPart(std::move(NdLPart));

          // Simple check to omit last non-terminal in substitution set.
          if (!matchPrefix(Ctx.Result.getMangledName(), State.Pos, "E")) {
            State.addSubstitute(NdNParts);
          }

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdNParts));

          State.advance();
        }
        break;
      // <nested-name--sfx> ::= ( ... ) ( () <nested-name--step>
      //                                  [() @ <template-args>] )* E
      case 8:
        Ctx.tryParseChildState(PSI_TemplateArgs, 10, false);
        break;
      // <nested-name--sfx> ::= ( ... ) ( () <nested-name--step>
      //                                  [() <template-args> @] )* E
      case 9:
        State.reset(false);
        State.moveToProduction(5, false);
        break;
      case 10:
        State.reset(false);
        State.moveToProduction(6, false);
        break;

      // <nested-name--sfx> ::= ( ... ) ( () <nested-name--step>
      //                                  [() <template-args>] )* @ E
      case 11:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "E")) {
          // <nested-name--sfx> ::= ( ... ) ( ... )* E @
          Ctx.reportSuccessToParent();
          break;
        }
        Ctx.reportFailToParent();
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;

    // MC***
    // <nested-name--start> ::= <unqualified-name>
    //                      ::= <template-param>
    //                      ::= <decltype>
    //
    // Result variables:   DmngRsltNode (same set as <substitution>)
    case PSI_NestedName_Start:
      switch (State.DotPos) {
      // <nested-name--start> ::= <unqualified-name>
      case 0:
        Ctx.tryParseChildState(PSI_UnqualifiedName, 2);
        break;
      // <nested-name--start> ::= <unqualified-name>
      case 1:
        Ctx.reportSuccessToParent();
        break;

      // <nested-name--start> ::= @ <template-param>
      case 2:
        Ctx.tryParseChildState(PSI_TemplateParam, 4);
        break;
      // <nested-name--start> ::= <template-param> @
      case 3:
        Ctx.reportSuccessToParent();
        break;

      // <prefix--start> ::= @ <decltype>
      case 4:
        Ctx.parseChildState(PSI_Decltype);
        break;
      // <prefix--start> ::= <decltype> @
      case 5:
        Ctx.reportSuccessToParent();
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;

    // MC***
    // <nested-name--step> ::= <data-member-prefix>
    //                     ::= <unqualified-name>
    //
    // Result variables:   DmngRsltNamePart
    case PSI_NestedName_Step:
      switch (State.DotPos) {
      // <nested-name--step> ::= @ <data-member-prefix>
      case 0:
        Ctx.tryParseChildState(PSI_DataMemberPrefix, 2);
        break;
      // <nested-name--step> ::= <data-member-prefix> @
      case 1:
        Ctx.reportSuccessToParent();
        break;


      // <nested-name--step> ::= @ <unqualified-name>
      case 2:
        Ctx.parseChildState(PSI_UnqualifiedName);
        break;
      // <nested-name--step> ::= <unqualified-name> @
      case 3:
        Ctx.reportSuccessToParent();
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <unscoped-name> ::= [St] <unqualified-name>
    //
    // Result variables:   DmngRsltName (DmngRsltOrdinaryName)
    case PSI_UnscopedName:
      switch (State.DotPos) {
      // <unscoped-name> ::= @ [St] <unqualified-name>
      case 0:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "St")) {
          // Assuming DNDK_NamePart node: "::std::"
          auto ElemIt = StdSubstitutions.find("St");
          if (ElemIt == StdSubstitutions.end()) {
            // ReSharper disable once CppUnreachableCode
            assert(false && "Missing standard substitution.");
            Ctx.reportFailToParent();
            break;
          }
          Ctx.Variables.emplace_back(ElemIt->second->clone());
        }
        State.advance();
        break;
      // <unscoped-name> ::= [St] @ <unqualified-name>
      case 1:
        Ctx.parseChildState(PSI_UnqualifiedName);
        break;
      // <unscoped-name> ::= [St] <unqualified-name> @
      case 2: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdNPart =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();

          auto NdOName = std::make_shared<DmngRsltOrdinaryName>(
            NdNPart->getAs<DNDK_NamePart>());

          for (std::size_t I = 1, E = Ctx.getStateVarsCount(); I < E; ++I) {
            const auto &NdNSubpart =
              Ctx.getStateVar(I).unsafeGetAsExact<ParseNodeT>();
            NdOName->addPart(NdNSubpart->getAs<DNDK_NamePart>());
          }

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdOName));

          Ctx.reportSuccessToParent();
        }
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <unscoped-template-name> ::= <unscoped-name>
    //                          ::= <substitution>
    //
    // Result variables:   DmngRsltName (DmngRsltOrdinaryName)
    case PSI_UnscopedTemplateName:
      switch (State.DotPos) {
      // <unscoped-template-name> ::= @ <unscoped-name>
      case 0:
        Ctx.tryParseChildState(PSI_UnscopedName, 2);
        break;
      // <unscoped-template-name> ::= <unscoped-name> @
      case 1: {
          INP_ENSURE_TMP_SIZE(1);

          const auto& NdOName =
              Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();

          State.addSubstituteNoClone(std::make_shared<DmngRsltNameParts>(
            *NdOName->getAs<DNDK_Name>()->getAs<DNK_Ordinary>()));

          Ctx.reportSuccessToParent();
        }
        break;

      // <unscoped-template-name> ::= @ <substitution>
      case 2:
        Ctx.parseChildState(PSI_Substitution);
        break;
      // <unscoped-template-name> ::= <substitution> @
      case 3: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdNode =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();
          auto NdNParts = adaptNode<DmngRsltNameParts>(NdNode);

          if (NdNParts == nullptr) {
            Ctx.reportFailToParent();
            break;
          }

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::make_shared<DmngRsltOrdinaryName>(
            NdNParts->getAs<DNDK_NameParts>()));

          Ctx.reportSuccessToParent();
        }
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <template-args> ::= I <template-arg>+ E
    //
    // Result variables:   DmngRsltTArg+
    case PSI_TemplateArgs:
      switch (State.DotPos) {
      // <template-args> ::= @ I <template-arg>+ E
      case 0:
        State.addTArgGroup();
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "I"))
          State.advance();
        else
          Ctx.reportFailToParent();
        break;
      // <template-args> ::= I @ <template-arg>+ E
      case 1:
        Ctx.parseChildState(PSI_TemplateArg);
        break;
      // <template-args> ::= I <template-arg>+ @ <template-arg>* E
      case 2:
        Ctx.tryParseChildState(PSI_TemplateArg, 4, false);
        break;
      case 3:
        // Flattening recursive production from step 2.
        State.reset(false);
        State.moveToProduction(2, false);
        break;
      // <template-args> ::= I <template-arg>+ @ E
      case 4:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "E"))
          State.advance();
        else
          Ctx.reportFailToParent();
        break;
      // <template-args> ::= I <template-arg>+ E @
      case 5:
        State.assignTArgGroupToScope();
        Ctx.reportSuccessToParent();
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <local-name> ::= Z <encoding> E <local-name--enty>
    // <local-name--enty> ::= <name> [<discriminator>]
    //                    ::= s [<discriminator>]
    //                    ::= d [ <number> ] _ <name>
    //
    // Result variables:   DmngRsltName (DmngRsltOrdinaryName)
    case PSI_LocalName:
      switch (State.DotPos) {
      // <local-name> ::= @ Z <encoding> E <local-name--enty>
      case 0:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "Z"))
          State.advance();
        else
          Ctx.reportFailToParent();
        break;
      // <local-name> ::= Z @ <encoding> E <local-name--enty>
      case 1:
        Ctx.parseChildState(PSI_Encoding);
        break;
      // <local-name> ::= Z <encoding> @ E <local-name--enty>
      case 2:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "E"))
          State.advance();
        else
          Ctx.reportFailToParent();
        break;
      // <local-name> ::= Z <encoding> E @ <local-name--enty>
      case 3:
        Ctx.parseChildState(PSI_LocalName_Enty);
        break;
      // <local-name> ::= Z <encoding> E <local-name--enty> @
      case 4: {
          // Type of local name (0 - local class, 1 - string literal,
          //                     2 - parameter scope).
          INP_ENSURE_TMP_SIZE(2);

          auto TypeSelIdx = Ctx.getStateVarsCount() - 1;

          auto NdScope = Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>()
            ->getAs<DNDK_Name>();
          auto TypeSel = Ctx.getStateVar(TypeSelIdx).unsafeGetAsExact<int>();

          // Local scope element.
          if (TypeSel == 0) {
            auto NdOName = Ctx.getStateVar(1).unsafeGetAsExact<ParseNodeT>()
              ->getAs<DNDK_Name>()->getAs<DNK_Ordinary>();

            NdOName->setLocalScope(std::move(NdScope));

            if (TypeSelIdx > 2) {
              auto Id = Ctx.getStateVar(2).unsafeGetAsExact<int>();
              NdOName->setInLocalScopeIdx(Id + 1);
            }

            Ctx.resetStateVars();
            Ctx.Variables.emplace_back(std::move(NdOName));

            Ctx.reportSuccessToParent();
            break;
          }
          // String literal.
          if (TypeSel == 1) {
            auto NdOName = std::make_shared<DmngRsltOrdinaryName>();

            NdOName->setLocalScope(std::move(NdScope));

            if (TypeSelIdx > 1) {
              auto Id = Ctx.getStateVar(1).unsafeGetAsExact<int>();
              NdOName->setInLocalScopeIdx(Id + 1);
            }

            Ctx.resetStateVars();
            Ctx.Variables.emplace_back(std::move(NdOName));

            Ctx.reportSuccessToParent();
            break;
          }
          // Default value of parameter.
          if (TypeSel == 2) {
            auto ParamIdx = Ctx.getStateVar(1).unsafeGetAsExact<int>();
            auto NdOName = Ctx.getStateVar(2).unsafeGetAsExact<ParseNodeT>()
              ->getAs<DNDK_Name>()->getAs<DNK_Ordinary>();

            NdOName->setLocalScope(std::move(NdScope));
            NdOName->setDefaultValueParamRIdx(ParamIdx);

            Ctx.resetStateVars();
            Ctx.Variables.emplace_back(std::move(NdOName));

            Ctx.reportSuccessToParent();
            break;
          }

          Ctx.reportFailToParent();
        }
        break;
      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;

    // MC***
    // <local-name--enty> ::= <name> [<discriminator>]
    //                    ::= s [<discriminator>]
    //                    ::= d [ <number> ] _ <name>
    case PSI_LocalName_Enty:
      switch (State.DotPos) {
      // <local-name--enty> ::= @ <name> [<discriminator>]
      case 0:
        Ctx.tryParseChildState(PSI_Name, 3);
        break;
      // <local-name--enty> ::= <name> @ [<discriminator>]
      case 1:
        Ctx.tryParseChildState(PSI_Discriminator, 2, false);
        break;
      // <local-name--enty> ::= <name> [<discriminator>] @
      case 2:
        Ctx.Variables.emplace_back(0);
        Ctx.reportSuccessToParent();
        break;

      // <local-name--enty> ::= @ s [<discriminator>]
      case 3:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "s"))
          State.advance();
        else
          State.moveToProduction(6);
        break;
      // <local-name--enty> ::= s @ [<discriminator>]
      case 4:
        Ctx.tryParseChildState(PSI_Discriminator, 5, false);
        break;
      // <local-name--enty> ::= s [<discriminator>] @
      case 5:
        Ctx.Variables.emplace_back(1);
        Ctx.reportSuccessToParent();
        break;

      // <local-name--enty> ::= @ d [ <number> ] _ <name>
      case 6:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "d"))
          State.advance();
        else
          Ctx.reportFailToParent();
        break;
      // <local-name--enty> ::= d @ [ <number> ] _ <name>
      case 7: {
          int ParamRIdx = 0;

          if (matchIntNumberAndAdvance<int, NMK_NonNegative>(
                Ctx.Result.getMangledName(), State.Pos, ParamRIdx)) {
            assert(ParamRIdx + 1 > ParamRIdx && "Param idx out of bounds.");
            ++ParamRIdx;
          }
          Ctx.Variables.emplace_back(ParamRIdx);
          State.advance();
        }
        break;
      // <local-name--enty> ::= d [ <number> ] @ _ <name>
      case 8:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "_"))
          State.advance();
        else
          Ctx.reportFailToParent();
        break;
      // <local-name--enty> ::= d [ <number> ] _ @ <name>
      case 9:
        Ctx.parseChildState(PSI_Name);
        break;
      // <local-name--enty> ::= d [ <number> ] _ <name> @
      case 10:
        Ctx.Variables.emplace_back(2);
        Ctx.reportSuccessToParent();
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <type> ::= <builtin-type>                    # before <class-enum-type>
    //        ::= <function-type>                   # before qual type
    //        ::= <class-enum-type>
    //        ::= <array-type>
    //        ::= <vector-type>
    //        ::= <pointer-to-member-type>
    //        ::= <template-template-param> <template-args>  #before <t-param>
    //        ::= <template-param>
    //        ::= <decltype>
    //        ::= P <type>
    //        ::= R <type>
    //        ::= O <type>
    //        ::= C <type>
    //        ::= G <type>
    //        ::= Dp <type>
    //        ::= ( <CV-qualifiers> | <vendor-qualifiers> )+ <type>
    //        ::= <substitution>
    //
    // Result variables:   DmngRsltType
    case PSI_Type:
      // Restrict scope of template parameters to type definition.
      if (!State.hasOwnTArgScope())
        State.addTArgScope();

      switch (State.DotPos) {
      // TODO: If special one-time preinitialization is no longer necessary,
      //       remove it in the future.
      case 0:
        State.advance();
        break;

      // <type> ::= @ <builtin-type>
      case 1:
        Ctx.tryParseChildState(PSI_BuiltinType, 3);
        break;
      // <type> ::= <builtin-type> @
      case 2: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdType =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();

          if (NdType->getAs<DNDK_Type>()->getAs<DTK_Builtin>()
                    ->isVendorBuiltinType())
            State.addSubstitute(NdType);

          Ctx.reportSuccessToParent();
        }
        break;

      // <type> ::= @ <function-type>
      case 3:
        Ctx.tryParseChildState(PSI_FunctionType, 5);
        break;
      // <type> ::= <function-type> @
      case 4: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdType =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();

          // Exception from substitution: function type when parent entity
          // is pointer to member is not candidate.
          auto CurrentState = State;
          Ctx.States.pop();

          if (Ctx.States.top().Id != PSI_PointerToMemberType)
            CurrentState.addSubstitute(NdType);

          Ctx.States.push(std::move(CurrentState));

          Ctx.reportSuccessToParent();
        }
        break;

      // <type> ::= @ <class-enum-type>
      case 5:
        Ctx.tryParseChildState(PSI_ClassEnumType, 7);
        break;
      // <type> ::= <class-enum-type> @
      case 6: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdType =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();

          State.addSubstitute(NdType);

          Ctx.reportSuccessToParent();
        }
        break;

      // <type> ::= @ <array-type>
      case 7:
        Ctx.tryParseChildState(PSI_ArrayType, 9);
        break;
      // <type> ::= <array-type> @
      case 8: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdType =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();

          State.addSubstitute(NdType);

          Ctx.reportSuccessToParent();
        }
        break;

      // <type> ::= @ <vector-type>
      case 9:
        Ctx.tryParseChildState(PSI_VectorType, 11);
        break;
      // <type> ::= <vector-type> @
      case 10: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdType =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();

          State.addSubstitute(NdType);

          Ctx.reportSuccessToParent();
        }
        break;

      // <type> ::= @ <pointer-to-member-type>
      case 11:
        Ctx.tryParseChildState(PSI_PointerToMemberType, 13);
        break;
      // <type> ::= <pointer-to-member-type> @
      case 12: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdType =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();

          State.addSubstitute(NdType);

          Ctx.reportSuccessToParent();
        }
        break;

      // <type> ::= @ <template-template-param> <template-args>
      case 13:
        Ctx.tryParseChildState(PSI_TemplateTemplateParam, 16);
        break;
      // <type> ::= <template-template-param> @ <template-args>
      case 14:
        Ctx.tryParseChildState(PSI_TemplateArgs, 16);
        break;
      // <type> ::= <template-template-param> <template-args> @
      case 15: {
          INP_ENSURE_TMP_SIZE(2);

          auto NdType = Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>()
            ->getAs<DNDK_Type>();

          DmngRsltTArgsBase *TArgsElem = nullptr;
          if (NdType->getKind() == DTK_TemplateParam)
            TArgsElem = NdType->getAs<DTK_TemplateParam>().get();
          if (NdType->getKind() == DTK_TypeName) {
            TArgsElem = const_cast<DmngRsltNamePart *>(
              NdType->getAs<DTK_TypeName>()->getTypeName()
                ->getLastPart().get());
          }

          if (TArgsElem == nullptr) {
            Ctx.reportFailToParent();
            break;
          }

          for (std::size_t I = 1, E = Ctx.getStateVarsCount(); I < E; ++I) {
            auto TArg = Ctx.getStateVar(I).unsafeGetAsExact<DmngRsltTArg>();
            TArgsElem->addTemplateArg(std::move(TArg));
          }

          State.addSubstitute(NdType);

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdType));

          Ctx.reportSuccessToParent();
        }
        break;

      // <type> ::= @ <template-param>
      case 16:
        Ctx.tryParseChildState(PSI_TemplateParam, 18);
        break;
      // <type> ::= <template-param> @
      case 17: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdExpr =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();
          auto NdType = std::make_shared<DmngRsltTParamType>(
            NdExpr->getAs<DNDK_Expr>()->getAs<DXK_TemplateParam>());

          State.addSubstitute(NdExpr);

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdType));

          Ctx.reportSuccessToParent();
        }
        break;

      // <type> ::= @ <decltype>
      case 18:
        Ctx.tryParseChildState(PSI_Decltype, 20);
        break;
      // <type> ::= <decltype> @
      case 19: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdExpr =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();
          auto NdType = std::make_shared<DmngRsltDecltypeType>(
            NdExpr->getAs<DNDK_Expr>()->getAs<DXK_Decltype>());

          State.addSubstitute(NdExpr);

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdType));

          Ctx.reportSuccessToParent();
        }
        break;

      // <type> ::= @ P <type>
      case 20:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "P"))
          State.advance();
        else
          State.moveToProduction(23);
        break;
      // <type> ::= P @ <type>
      case 21:
        Ctx.tryParseChildState(PSI_Type, 23);
        break;
      // <type> ::= P <type> @
      case 22: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdType =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();
          auto NdQType = std::make_shared<DmngRsltQualType>(
            DmngRsltQualType::createPointer(NdType->getAs<DNDK_Type>()));

          State.addSubstitute(NdQType);

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdQType));

          Ctx.reportSuccessToParent();
        }
        break;

      // <type> ::= @ R <type>
      case 23:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "R"))
          State.advance();
        else
          State.moveToProduction(26);
        break;
      // <type> ::= R @ <type>
      case 24:
        Ctx.tryParseChildState(PSI_Type, 26);
        break;
      // <type> ::= R <type> @
      case 25: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdType =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();
          auto NdQType = std::make_shared<DmngRsltQualType>(
            DmngRsltQualType::createLValueRef(NdType->getAs<DNDK_Type>()));

          State.addSubstitute(NdQType);

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdQType));

          Ctx.reportSuccessToParent();
        }
        break;

      // <type> ::= @ O <type>
      case 26:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "O"))
          State.advance();
        else
          State.moveToProduction(29);
        break;
      // <type> ::= O @ <type>
      case 27:
        Ctx.tryParseChildState(PSI_Type, 29);
        break;
      // <type> ::= O <type> @
      case 28: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdType =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();
          auto NdQType = std::make_shared<DmngRsltQualType>(
            DmngRsltQualType::createRValueRef(NdType->getAs<DNDK_Type>()));

          State.addSubstitute(NdQType);

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdQType));

          Ctx.reportSuccessToParent();
        }
        break;

      // <type> ::= @ C <type>
      case 29:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "C"))
          State.advance();
        else
          State.moveToProduction(32);
        break;
      // <type> ::= C @ <type>
      case 30:
        Ctx.tryParseChildState(PSI_Type, 32);
        break;
      // <type> ::= C <type> @
      case 31: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdType =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();
          auto NdQType = std::make_shared<DmngRsltQualType>(
            DmngRsltQualType::createComplex(NdType->getAs<DNDK_Type>()));

          State.addSubstitute(NdQType);

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdQType));

          Ctx.reportSuccessToParent();
        }
        break;

      // <type> ::= @ G <type>
      case 32:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "G"))
          State.advance();
        else
          State.moveToProduction(35);
        break;
      // <type> ::= G @ <type>
      case 33:
        Ctx.tryParseChildState(PSI_Type, 35);
        break;
      // <type> ::= G <type> @
      case 34: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdType =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();
          auto NdQType = std::make_shared<DmngRsltQualType>(
            DmngRsltQualType::createImaginary(NdType->getAs<DNDK_Type>()));

          State.addSubstitute(NdQType);

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdQType));

          Ctx.reportSuccessToParent();
        }
        break;

      // <type> ::= @ Dp <type>
      case 35:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "Dp"))
          State.advance();
        else
          State.moveToProduction(38);
        break;
      // <type> ::= Dp @ <type>
      case 36:
        Ctx.tryParseChildState(PSI_Type, 38);
        break;
      // <type> ::= Dp <type> @
      case 37: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdType =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();
          auto NdQType = std::make_shared<DmngRsltQualType>(
            DmngRsltQualType::createPackExpansion(
              NdType->getAs<DNDK_Type>()));

          State.addSubstitute(NdQType);

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdQType));

          Ctx.reportSuccessToParent();
        }
        break;

      // <type> ::= ( @ <CV-qualifiers> | <vendor-qualifiers> )+ <type>
      case 38: {
          DmngCvrQuals CvrQuals;
          if (matchCvrQualsAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, CvrQuals)) {
              Ctx.Variables.emplace_back(CvrQuals);
              State.advance();
          }
          else
            State.moveToProduction(42);
        }
        break;
      // <type> ::= ( <CV-qualifiers> | <vendor-qualifiers> )+
      //            ( @ <CV-qualifiers> | <vendor-qualifiers> )* <type>
      case 39: {
          DmngCvrQuals CvrQuals;
          if (matchCvrQualsAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, CvrQuals)) {
              Ctx.Variables.emplace_back(CvrQuals);
          }
          else
            State.advance();
        }
        break;
      // <type> ::= ( <CV-qualifiers> | <vendor-qualifiers> )+
      //            ( <CV-qualifiers> | @ <vendor-qualifiers> )* <type>
      case 40:
        Ctx.tryParseChildState(PSI_VendorQuals, 44, false);
        break;
      case 41:
        // Flattening recursive production.
        State.reset(false);
        State.moveToProduction(39, false);
        break;
      // <type> ::= ( <CV-qualifiers> | @ <vendor-qualifiers> )+ <type>
      case 42:
        Ctx.tryParseChildState(PSI_VendorQuals, 46);
        break;
      case 43:
        // Flattening recursive production.
        State.reset(false);
        State.moveToProduction(39, false);
        break;
      // <type> ::= ( <CV-qualifiers> | <vendor-qualifiers> )+ @ <type>
      case 44:
        Ctx.tryParseChildState(PSI_Type, 46);
        break;
      // <type> ::= ( <CV-qualifiers> | <vendor-qualifiers> )+ <type> @
      case 45: {
          INP_ENSURE_TMP_SIZE(2);

          auto QualsCount = Ctx.getStateVarsCount() - 1;

          const auto &NdType =
            Ctx.getStateVar(QualsCount).unsafeGetAsExact<ParseNodeT>();
          auto NdQType = std::make_shared<DmngRsltQualGrpType>(
            NdType->getAs<DNDK_Type>());

          DmngCvrQuals CvrQuals = DCVQ_None;
          for (std::size_t I = 0; I < QualsCount; ++I) {
            const auto &Qual = Ctx.getStateVar(I);

            if (Qual.isAExact<DmngRsltVendorQual>()) {
              const auto &VQual = Qual.unsafeGetAsExact<DmngRsltVendorQual>();
              auto ASQual = ASExtractFunc(VQual);

              if (ASQual != DASQ_None)
                NdQType->setAsQuals(ASQual);
              NdQType->addVendorQual(std::move(VQual));
            }
            else if(Qual.isAExact<DmngCvrQuals>())
              CvrQuals = CvrQuals | Qual.unsafeGetAsExact<DmngCvrQuals>();
          }
          NdQType->setCvrQuals(CvrQuals);

          State.addSubstitute(NdQType);

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdQType));

          Ctx.reportSuccessToParent();
        }
        break;

      // <type> ::= @ <substitution>
      case 46:
        Ctx.parseChildState(PSI_Substitution);
        break;
      // <type> ::= <substitution> @
      case 47: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdNode =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();
          auto NdType = adaptNode<DmngRsltType>(NdNode);

          if (NdType == nullptr) {
            Ctx.reportFailToParent();
            break;
          }

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdType));

          Ctx.reportSuccessToParent();
        }
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <call-offset> ::= h <nv-offset> _
    //               ::= v <v-offset> _
    // <nv-offset> ::= <number>
    // <v-offset>  ::= <number> _ <number>
    //
    // Result variables:   DmngRsltAdjustOffset
    case PSI_CallOffset:
      switch (State.DotPos) {
      // <call-offset> ::= @ h <number> _
      case 0: {
          if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "h")) {
            State.moveToProduction(1);
            break;
          }

          // <call-offset> ::= h @ <number> _
          unsigned long long Offset;
          if (!matchIntNumberAndAdvance(Ctx.Result.getMangledName(),
                                        State.Pos, Offset)) {
            State.moveToProduction(1);
            break;
          }

          // <call-offset> ::= h <number> @ _
          if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "_")) {
            State.moveToProduction(1);
            break;
          }

          // <call-offset> ::= h <number> _ @
          Ctx.Variables.emplace_back(DmngRsltAdjustOffset(Offset));
          Ctx.reportSuccessToParent();
        }
        break;

      // <call-offset> ::= @ v <number> _ <number> _
      case 1: {
          if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "v")) {
            Ctx.reportFailToParent();
            break;
          }

          // <call-offset> ::= v @ <number> _ <number> _
          unsigned long long BaseOffset;
          if (!matchIntNumberAndAdvance(Ctx.Result.getMangledName(),
                                        State.Pos, BaseOffset)) {
            Ctx.reportFailToParent();
            break;
          }

          // <call-offset> ::= v <number> @ _ <number> _
          if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "_")) {
            Ctx.reportFailToParent();
            break;
          }

          // <call-offset> ::= v <number> _ @ <number> _
          unsigned long long VCallOffset;
          if (!matchIntNumberAndAdvance(Ctx.Result.getMangledName(),
                                        State.Pos, VCallOffset)) {
            Ctx.reportFailToParent();
            break;
          }

          // <call-offset> ::= v <number> _ <number> @ _
          if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "_")) {
            Ctx.reportFailToParent();
            break;
          }

          // <call-offset> ::= v <number> _ <number> _ @
          Ctx.Variables.emplace_back(
            DmngRsltAdjustOffset(BaseOffset, VCallOffset));
          Ctx.reportSuccessToParent();
        }
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <vendor-qualifiers> ::= <vendor-qualifier>+
    //
    // Result variables:   DmngRsltVendorQual+
    case PSI_VendorQuals:
      switch (State.DotPos) {
      // <vendor-qualifiers> ::= @ <vendor-qualifier>+
      case 0:
        Ctx.parseChildState(PSI_VendorQual);
        break;
      // <vendor-qualifiers> ::= <vendor-qualifier>+ @ <vendor-qualifier>*
      case 1:
        Ctx.tryParseChildState(PSI_VendorQual, 3, false);
        break;
      case 2:
        // Flattening recursive production.
        State.reset(false);
        State.moveToProduction(1, false);
        break;
      // <vendor-qualifiers> ::= <vendor-qualifier>+ @
      case 3:
        Ctx.reportSuccessToParent();
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <unqualified-name> ::= <operator-name>
    //                    ::= <ctor-dtor-name>
    //                    ::= <source-name>
    //                    ::= <unnamed-type-name>
    //
    // Result variables:   DmngRsltNamePart
    case PSI_UnqualifiedName:
      switch (State.DotPos) {
      // <unqualified-name> ::= @ <operator-name>
      case 0:
        Ctx.tryParseChildState(PSI_OperatorName, 2);
        break;
      // <unqualified-name> ::= <operator-name> @
      case 1:
        Ctx.reportSuccessToParent();
        break;

      // <unqualified-name> ::= @ <ctor-dtor-name>
      case 2: {
          bool IsCtor = false;
          DmngCtorDtorType CtorDtorType;
          if (matchCtorDtorAndAdvance(Ctx.Result.getMangledName(), State.Pos,
                                      IsCtor, CtorDtorType)) {
            // <unqualified-name> ::= <ctor-dtor-name> @
            Ctx.Variables.emplace_back(
              std::make_shared<DmngRsltCtorDtorNamePart>(IsCtor,
                                                          CtorDtorType));

            Ctx.reportSuccessToParent();
            break;
          }

          State.moveToProduction(3);
        }
        break;

      // <unqualified-name> ::= @ <source-name>
      case 3: {
          std::string SrcName;
          if (matchSourceNameAndAdvance(Ctx.Result.getMangledName(),
                                        State.Pos, SrcName)) {
            // <unqualified-name> ::= <source-name> @
            Ctx.Variables.emplace_back(
              std::make_shared<DmngRsltSrcNamePart>(SrcName));

            Ctx.reportSuccessToParent();
            break;
          }

          State.moveToProduction(4);
        }
        break;

      // <unqualified-name> ::= @ <unnamed-type-name>
      case 4:
        Ctx.parseChildState(PSI_UnnamedTypeName);
        break;
      // <unqualified-name> ::= <unnamed-type-name> @
      case 5:
        Ctx.reportSuccessToParent();
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <substitution> ::= S [<seq-id>] _
    //                ::= <std-sub>
    //
    // Result variables:   DmngRsltNode (DmngRsltNamePart | DmngRsltNameParts
    //                     | DmngRsltTParamExpr | DmngRsltDecltypeExpr
    //                     | DmngRsltType )
    case PSI_Substitution:
      switch (State.DotPos) {
      // <substitution> ::= @ S [<seq-id>] _
      case 0: {
          if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "S")) {
            State.moveToProduction(1);
            break;
          }

          // <substitution> ::= S @ [<seq-id>] _
          ParseContext::SubsT::size_type SubId = 0;
          if (matchSeqIdAndAdvance(Ctx.Result.getMangledName(),
                                    State.Pos, SubId)) {
            assert(SubId + 1 > SubId && "Id out of bounds.");
            ++SubId;
          }

          // <substitution> ::= S [<seq-id>] @ _
          if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "_")) {
            // <substitution> ::= S [<seq-id>] _ @
            if (SubId >= Ctx.Substitutions.size()) {
              // ReSharper disable once CppUnreachableCode
              assert(false && "Missing substitution.");
              State.moveToProduction(1);
              break;
            }

            // Direct substitutions for template parameters need to be
            // registered for referencing.
            auto NdTParamExpr = createUnwrappedTParamExpr(
              Ctx.Substitutions[SubId].second);
            if (NdTParamExpr != nullptr) {
              State.addTParam(NdTParamExpr);
              Ctx.Variables.emplace_back(std::move(NdTParamExpr));

              Ctx.reportSuccessToParent();
              break;
            }

            Ctx.Variables.emplace_back(
              Ctx.Substitutions[SubId].second->clone());

            Ctx.reportSuccessToParent();
            break;
          }

          State.moveToProduction(1);
        }
        break;

      // <substitution> ::= @ <std-sub>
      case 1: {
          std::string StdSubId;
          if (matchStdSubAndAdvance(Ctx.Result.getMangledName(),
                                    State.Pos, StdSubId)) {
            // <substitution> ::= <std-sub> @
            auto ElemIt = StdSubstitutions.find(StdSubId);
            if (ElemIt == StdSubstitutions.end()) {
              // ReSharper disable once CppUnreachableCode
              assert(false && "Missing standard substitution.");
              Ctx.reportFailToParent();
              break;
            }

            Ctx.Variables.emplace_back(ElemIt->second->clone());

            Ctx.reportSuccessToParent();
            break;
          }

          Ctx.reportFailToParent();
        }
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <template-arg> ::= <type>
    //                ::= X <expression> E
    //                ::= <expr-primary>
    //                ::= J <template-arg>* E
    //
    // Result variables:   DmngRsltTArg
    case PSI_TemplateArg:
    case PSI_TemplateArg_Pack:
      switch (State.DotPos) {
      // <template-arg> ::= @ <type>
      case 0:
        Ctx.tryParseChildState(PSI_Type, 2);
        break;
      // <template-arg> ::= <type> @
      case 1: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdType =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();

          DmngRsltTArg Arg(NdType->getAs<DNDK_Type>());

          Ctx.resetStateVars();
          if (State.Id != PSI_TemplateArg_Pack)
            State.addTArg(std::make_shared<DmngRsltTArg>(Arg));
          Ctx.Variables.emplace_back(std::move(Arg));

          Ctx.reportSuccessToParent();
        }
        break;

      // <template-arg> ::= @ X <expression> E
      case 2:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "X")) {
          State.advance();
          break;
        }
        State.moveToProduction(5);
        break;
      // <template-arg> ::= X @ <expression> E
      case 3:
        Ctx.tryParseChildState(PSI_Expression, 5);
        break;
      // <template-arg> ::= X <expression> @ E
      case 4:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "E")) {
          // <template-arg> ::= X <expression> E @
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdExpr =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();

          DmngRsltTArg Arg(NdExpr->getAs<DNDK_Expr>());

          Ctx.resetStateVars();
          if (State.Id != PSI_TemplateArg_Pack)
            State.addTArg(std::make_shared<DmngRsltTArg>(Arg));
          Ctx.Variables.emplace_back(std::move(Arg));

          Ctx.reportSuccessToParent();
          break;
        }
        State.moveToProduction(5);
        break;

      // <template-arg> ::= @ <expr-primary>
      case 5:
        Ctx.tryParseChildState(PSI_ExprPrimary, 7);
        break;
      // <template-arg> ::= <expr-primary> @
      case 6: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdPExpr =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();

          DmngRsltTArg Arg(NdPExpr->getAs<DNDK_Expr>());

          Ctx.resetStateVars();
          if (State.Id != PSI_TemplateArg_Pack)
            State.addTArg(std::make_shared<DmngRsltTArg>(Arg));
          Ctx.Variables.emplace_back(std::move(Arg));

          Ctx.reportSuccessToParent();
        }
        break;

      // <template-arg> ::= @ J <template-arg>* E
      case 7:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "J")) {
          State.advance();
          break;
        }
        Ctx.reportFailToParent();
        break;
      // <template-arg> ::= J <template-arg>* @ <template-arg>* E
      case 8:
        Ctx.tryParseChildState(PSI_TemplateArg_Pack, 10, false);
        break;
      case 9:
        // Flattening recursive production.
        State.reset(false);
        State.moveToProduction(8, false);
        break;
      // <template-arg> ::= J <template-arg>* @ E
      case 10:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "E")) {
          // <template-arg> ::= J <template-arg>* E @
          DmngRsltTArg Arg;
          for (std::size_t I = 0, E = Ctx.getStateVarsCount(); I < E; ++I) {
            const auto &PackArg =
              Ctx.getStateVar(I).unsafeGetAsExact<DmngRsltTArg>();

            Arg.addPackArg(PackArg);
          }

          Ctx.resetStateVars();
          if (State.Id != PSI_TemplateArg_Pack)
            State.addTArg(std::make_shared<DmngRsltTArg>(Arg));
          Ctx.Variables.emplace_back(std::move(Arg));

          Ctx.reportSuccessToParent();
          break;
        }
        Ctx.reportFailToParent();
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <discriminator> ::= __ <number> _
    //                 ::= _ <number> # two characters
    //
    // Result variables:   int
    case PSI_Discriminator:
      switch (State.DotPos) {
      // <discriminator> ::= @ __ <number> _
      case 0: {
          if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "__")) {
            State.moveToProduction(1);
            break;
          }

          // <discriminator> ::= __ @ <number> _
          int DiscriminatorIdx = 0;
          if (!matchIntNumberAndAdvance<int, NMK_NonNegative>(
                Ctx.Result.getMangledName(), State.Pos, DiscriminatorIdx)) {
            State.moveToProduction(1);
            break;
          }

          // <discriminator> ::= __ <number> @ _
          if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                    State.Pos, "_")) {
            // <discriminator> ::= __ <number> _ @
            Ctx.Variables.emplace_back(DiscriminatorIdx);

            Ctx.reportSuccessToParent();
            break;
          }

          State.moveToProduction(1);
        }
        break;

      // <discriminator> ::= @ _ <number> # two characters
      case 1: {
          if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "_")) {
            Ctx.reportFailToParent();
            break;
          }

          // <discriminator> ::=  _ @ <number> # two characters
          int DiscriminatorIdx = 0;
          if (matchIntNumberAndAdvance<int, NMK_NonNegative, 1>(
                Ctx.Result.getMangledName(), State.Pos, DiscriminatorIdx)) {
            // <discriminator> ::=  _ <number> @ # two characters
            Ctx.Variables.emplace_back(DiscriminatorIdx);

            Ctx.reportSuccessToParent();
            break;
          }

          Ctx.reportFailToParent();
        }
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <builtin-type> ::= <fixed-builtin-type>
    //                ::= u <source-name>
    //
    // Result variables:   DmngRsltType (DmngRsltBuiltinType)
    case PSI_BuiltinType:
      switch (State.DotPos) {
      // <builtin-type> ::= @ <fixed-builtin-type>
      case 0: {
          DmngBuiltinType FixedType = DBT_Void;
          if (matchFixedBiTypeAndAdvance(Ctx.Result.getMangledName(),
                                          State.Pos, FixedType)) {
            // <builtin-type> ::= <fixed-builtin-type> @
            Ctx.Variables.emplace_back(
              std::make_shared<DmngRsltBuiltinType>(FixedType));

            Ctx.reportSuccessToParent();
            break;
          }
          State.moveToProduction(1);
        }
        break;

      // <builtin-type> ::= @ u <source-name>
      case 1: {
          if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "u")) {
            Ctx.reportFailToParent();
            break;
          }

          // <builtin-type> ::= u @ <source-name>
          std::string VendorType;
          if (matchSourceNameAndAdvance(Ctx.Result.getMangledName(),
                                        State.Pos, VendorType)) {
            // <builtin-type> ::= u <source-name> @
            Ctx.Variables.emplace_back(
              std::make_shared<DmngRsltBuiltinType>(VendorType));

            Ctx.reportSuccessToParent();
            break;
          }

          Ctx.reportFailToParent();
        }
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <function-type> ::= [<vendor-qualifiers>] [<CV-qualifiers>]
    //                     [<vendor-qualifiers>] F [Y] <bare-function-type>
    //                     [<ref-qualifier>] E
    //
    // Result variables:   DmngRsltType (DmngRsltFuncType)
    case PSI_FunctionType:
      switch (State.DotPos) {
      case 0:
        // Initializing helper variables.
        Ctx.Variables.emplace_back(); // Position of first type.
        State.advance();
        break;
      // <function-type> ::= @ [<vendor-qualifiers>] [<CV-qualifiers>] ...
      case 1:
        Ctx.tryParseChildState(PSI_VendorQuals, 2, false);
        break;
      // <function-type> ::= [<vendor-qualifiers>] @ [<CV-qualifiers>] ...
      case 2: {
          DmngCvrQuals CvrQuals = DCVQ_None;
          if (matchCvrQualsAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, CvrQuals)) {
            Ctx.Variables.emplace_back(CvrQuals);
          }
          State.advance();
        }
        break;
      // <function-type> ::= ... [<CV-qualifiers>] @ [<vendor-qualifiers>] ...
      case 3:
        Ctx.tryParseChildState(PSI_VendorQuals, 4, false);
        break;
      // <function-type> ::= ... [<vendor-qualifiers>] @ F [Y] ...
      case 4: {
          if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "F")) {
            Ctx.reportFailToParent();
            break;
          }

          // <function-type> ::= ... [<vendor-qualifiers>] F @ [Y] ...
          bool IsExternC = matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                                  State.Pos, "Y");
          Ctx.Variables.emplace_back(IsExternC);
          Ctx.getStateVar(0) = Ctx.getStateVarsCount();
          State.advance();
        }
        break;
      // <function-type> ::=... [Y] @ <bare-function-type> [<ref-qualifier>] E
      case 5:
        Ctx.parseChildState(PSI_BareFunctionType);
        break;
      // <function-type> ::= ... <bare-function-type> @ [<ref-qualifier>] E
      case 6: {
          DmngRefQuals RefQuals = DRQ_None;
          matchRefQualsAndAdvance(Ctx.Result.getMangledName(), State.Pos,
                                  RefQuals);

          // <function-type> ::= ... [<ref-qualifier>] @ E
          if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "E")) {
            Ctx.reportFailToParent();
            break;
          }

          // <function-type> ::= ... [<ref-qualifier>] E @
          INP_ENSURE_TMP_SIZE(3);
          auto SignatureStart = Ctx.getStateVar(0).unsafeGetAs<std::size_t>();
          INP_ENSURE_TMP_SIZE(SignatureStart + 1);

          const auto &NdArg0Type =
            Ctx.getStateVar(SignatureStart).unsafeGetAsExact<ParseNodeT>();

          auto NdFuncType = std::make_shared<DmngRsltFuncType>(
            NdArg0Type->getAs<DNDK_Type>());

          for (std::size_t I = SignatureStart + 1,
                            E = Ctx.getStateVarsCount(); I < E; ++I) {
            const auto &NdArgType =
              Ctx.getStateVar(I).unsafeGetAsExact<ParseNodeT>();
            NdFuncType->addSignatureType(NdArgType->getAs<DNDK_Type>());
          }

          for (std::size_t I = 0; I < SignatureStart - 1; ++I) {
            const auto &Qual = Ctx.getStateVar(I);

            if (Qual.isAExact<DmngRsltVendorQual>()) {
              const auto &VQual = Qual.unsafeGetAsExact<DmngRsltVendorQual>();
              auto ASQual = ASExtractFunc(VQual);

              if (ASQual != DASQ_None)
                NdFuncType->setAsQuals(ASQual);
              NdFuncType->addVendorQual(VQual);
            }
            else if(Qual.isAExact<DmngCvrQuals>())
              NdFuncType->setCvrQuals(Qual.unsafeGetAsExact<DmngCvrQuals>());
          }

          NdFuncType->setRefQuals(RefQuals);
          NdFuncType->setIsExternC(
            Ctx.getStateVar(SignatureStart - 1).unsafeGetAsExact<bool>());

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdFuncType));

          Ctx.reportSuccessToParent();
        }
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <class-enum-type> ::= <name>
    //                   ::= Ts <name>
    //                   ::= Tu <name>
    //                   ::= Te <name>
    //
    // Result variables:   DmngRsltType (DmngRsltTypeNameType)
    case PSI_ClassEnumType:
      switch (State.DotPos) {
      // <class-enum-type> ::= @ [ Ts | Tu | Te ] <name>
      case 0:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "Ts")) {
          Ctx.Variables.emplace_back(DTNK_ElaboratedClass);
          State.advance();
          break;
        }
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "Tu")) {
          Ctx.Variables.emplace_back(DTNK_ElaboratedUnion);
          State.advance();
          break;
        }
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "Te")) {
          Ctx.Variables.emplace_back(DTNK_ElaboratedEnum);
          State.advance();
          break;
        }
        Ctx.Variables.emplace_back(DTNK_None);
        State.advance();
        break;
      // <class-enum-type> ::= [ Ts | Tu | Te ] @ <name>
      case 1:
        Ctx.parseChildState(PSI_Name);
        break;
      // <class-enum-type> ::= [ Ts | Tu | Te ] <name> @
      case 2: {
          INP_ENSURE_TMP_SIZE(2);

          auto ElaborationType =
            Ctx.getStateVar(0).unsafeGetAsExact<DmngTypeNameKind>();
          const auto &NdOName =
            Ctx.getStateVar(1).unsafeGetAsExact<ParseNodeT>();

          auto NdTypeName = std::make_shared<DmngRsltTypeNameType>(
            NdOName->getAs<DNDK_Name>()->getAs<DNK_Ordinary>(),
            ElaborationType);

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdTypeName));

          Ctx.reportSuccessToParent();
        }
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <array-type> ::= A <number> _ <type>
    //              ::= A [<expression>] _ <type>
    //
    // Result variables:   DmngRsltType (DmngRsltArrayVecType)
    case PSI_ArrayType:
      switch (State.DotPos) {
      // <array-type> ::= @ A [<number> | <expression>] _ <type>
      case 0: {
          if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "A")) {
            Ctx.reportFailToParent();
            break;
          }

          // <array-type> ::= A @ [<number>] _ <type>
          unsigned long long ElemsCount = 0;
          if (matchIntNumberAndAdvance<unsigned long long, NMK_Positive>(
                Ctx.Result.getMangledName(), State.Pos, ElemsCount)) {
            Ctx.Variables.emplace_back(ElemsCount);
            State.moveToProduction(2, false);
            break;
          }
          State.advance();
        }
        break;
      // <array-type> ::= A @ [<expression>] _ <type>
      case 1:
        Ctx.tryParseChildState(PSI_Expression, 2, false);
        break;
      // <array-type> ::= A [<number> | <expression>] @ _ <type>
      case 2:
        if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                    State.Pos, "_")) {
          Ctx.reportFailToParent();
          break;
        }
        State.advance();
        break;
      // <array-type> ::= A [<number> | <expression>] _ @ <type>
      case 3:
        Ctx.parseChildState(PSI_Type);
        break;
      // <array-type> ::= A [<number> | <expression>] _ <type> @
      case 4: {
          INP_ENSURE_TMP_SIZE(1);

          ParseNodeT NdArrayType;
          if (Ctx.getStateVarsCount() == 1) {
            const auto &NdElemType =
              Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();

            NdArrayType = std::make_shared<DmngRsltArrayVecType>(
              NdElemType->getAs<DNDK_Type>());
          }
          else {
            const auto &Dims = Ctx.getStateVar(0);
            const auto &NdElemType =
              Ctx.getStateVar(1).unsafeGetAsExact<ParseNodeT>();

            if (Dims.isAExact<ParseNodeT>()) {
              NdArrayType = std::make_shared<DmngRsltArrayVecType>(
                NdElemType->getAs<DNDK_Type>(),
                Dims.unsafeGetAsExact<ParseNodeT>()->getAs<DNDK_Expr>());
            }
            else {
              NdArrayType = std::make_shared<DmngRsltArrayVecType>(
                NdElemType->getAs<DNDK_Type>(),
                Dims.unsafeGetAsExact<unsigned long long>());
            }
          }

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdArrayType));

          Ctx.reportSuccessToParent();
        }
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <vector-type> ::= Dv <number> _ <vec-ext-type>
    //               ::= Dv [<expression>] _ <vec-ext-type>
    // <vec-ext-type> ::= <type>
    //                ::= p
    //
    // Result variables:   DmngRsltType (DmngRsltArrayVecType)
    case PSI_VectorType:
      switch (State.DotPos) {
      // <vector-type> ::= @ Dv [<number> | <expression>] _ <vec-ext-type>
      case 0: {
          if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "Dv")) {
            Ctx.reportFailToParent();
            break;
          }

          // <vector-type> ::= Dv @ [<number>] _ <vec-ext-type>
          unsigned long long ElemsCount = 0;
          if (matchIntNumberAndAdvance<unsigned long long, NMK_Positive>(
                Ctx.Result.getMangledName(), State.Pos, ElemsCount)) {
            Ctx.Variables.emplace_back(ElemsCount);
            State.moveToProduction(2, false);
            break;
          }
          State.advance();
        }
        break;
      // <vector-type> ::= Dv @ [<expression>] _ <vec-ext-type>
      case 1:
        Ctx.tryParseChildState(PSI_Expression, 2, false);
        break;
      // <vector-type> ::= Dv [<number> | <expression>] @ _ <vec-ext-type>
      case 2:
        if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                    State.Pos, "_")) {
          Ctx.reportFailToParent();
          break;
        }

        // <vector-type> ::= Dv [<number> | <expression>] _ @ p
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "p")) {
          Ctx.Variables.emplace_back(
            std::make_shared<DmngRsltBuiltinType>(DBT_Pixel));
          State.moveToProduction(4, false);
          break;
        }
        State.advance();
        break;
      // <vector-type> ::= Dv [<number> | <expression>] _ @ <type>
      case 3:
        Ctx.parseChildState(PSI_Type);
        break;
      // <array-type> ::= A [<number> | <expression>] _ <vec-ext-type> @
      case 4: {
          INP_ENSURE_TMP_SIZE(1);

          ParseNodeT NdVecType;
          if (Ctx.getStateVarsCount() == 1) {
            const auto &NdElemType =
              Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();

            NdVecType = std::make_shared<DmngRsltArrayVecType>(
              NdElemType->getAs<DNDK_Type>(), true);
          }
          else {
            const auto &Dims = Ctx.getStateVar(0);
            const auto &NdElemType =
              Ctx.getStateVar(1).unsafeGetAsExact<ParseNodeT>();

            if (Dims.isAExact<ParseNodeT>()) {
              NdVecType = std::make_shared<DmngRsltArrayVecType>(
                NdElemType->getAs<DNDK_Type>(),
                Dims.unsafeGetAsExact<ParseNodeT>()->getAs<DNDK_Expr>(),
                true);
            }
            else {
              NdVecType = std::make_shared<DmngRsltArrayVecType>(
                NdElemType->getAs<DNDK_Type>(),
                Dims.unsafeGetAsExact<unsigned long long>(), true);
            }
          }

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdVecType));

          Ctx.reportSuccessToParent();
        }
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <pointer-to-member-type> ::= M <type> <type>
    //
    // Result variables:   DmngRsltType (DmngRsltPtr2MmbrType)
    case PSI_PointerToMemberType:
      switch (State.DotPos) {
      // <pointer-to-member-type> ::= @ M <type> <type>
      case 0:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "M")) {
          State.advance();
          break;
        }
        Ctx.reportFailToParent();
        break;
      // <pointer-to-member-type> ::= M @ <type> <type>
      case 1:
        Ctx.parseChildState(PSI_Type);
        break;
      // <pointer-to-member-type> ::= M <type> @ <type>
      case 2:
        Ctx.parseChildState(PSI_Type);
        break;
      // <pointer-to-member-type> ::= M <type> <type> @
      case 3: {
          INP_ENSURE_TMP_SIZE(2);

          const auto &NdClassType =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();
          const auto &NdMemberType =
            Ctx.getStateVar(1).unsafeGetAsExact<ParseNodeT>();

          auto NdPtr2MembType = std::make_shared<DmngRsltPtr2MmbrType>(
            NdClassType->getAs<DNDK_Type>(),
            NdMemberType->getAs<DNDK_Type>());

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdPtr2MembType));

          Ctx.reportSuccessToParent();
        }
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <template-param> ::= T [<number>] _
    //
    // Result variables:   DmngRsltExpr (DmngRsltTParamExpr)
    case PSI_TemplateParam:
      switch (State.DotPos) {
      // <template-param> ::= @ T [<number>] _
      case 0: {
          if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "T")) {
            Ctx.reportFailToParent();
            break;
          }

          // <template-param> ::= T @ [<number>] _
          unsigned ArgIdx = 0;
          if (matchIntNumberAndAdvance<unsigned, NMK_NonNegative>(
                Ctx.Result.getMangledName(), State.Pos, ArgIdx)) {
            assert(ArgIdx + 1 > ArgIdx && "Argument index is out of bounds.");
            ++ArgIdx;
          }

          // <template-param> ::= T [<number>] @ _
          if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "_")) {
            Ctx.reportFailToParent();
            break;
          }

          // <template-param> ::= T [<number>] _ @
          auto NdTParam = std::make_shared<DmngRsltTParamExpr>(ArgIdx);

          State.addTParam(NdTParam);
          Ctx.Variables.emplace_back(std::move(NdTParam));

          Ctx.reportSuccessToParent();
        }
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <template-template-param> ::= <template-param>
    //                           ::= <substitution>
    //
    // Result variables:   DmngRsltType
    case PSI_TemplateTemplateParam:
      switch (State.DotPos) {
      // <template-template-param> ::= @ <template-param>
      case 0:
        Ctx.tryParseChildState(PSI_TemplateParam, 2);
        break;
      // <template-template-param> ::= <template-param> @
      case 1: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdExpr =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();
          auto NdType = std::make_shared<DmngRsltTParamType>(
            NdExpr->getAs<DNDK_Expr>()->getAs<DXK_TemplateParam>());

          State.addSubstitute(NdExpr);

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdType));

          Ctx.reportSuccessToParent();
        }
        break;

      // <template-template-param> ::= @ <substitution>
      case 2:
        Ctx.parseChildState(PSI_Substitution);
        break;
      // <template-template-param> ::= <substitution> @
      case 3: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdNode =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();
          auto NdType = adaptNode<DmngRsltType>(NdNode);

          if (NdType == nullptr) {
            Ctx.reportFailToParent();
            break;
          }

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdType));

          Ctx.reportSuccessToParent();
        }
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <decltype> ::= Dt <expression> E
    //            ::= DT <expression> E
    //
    // Result variables:   DmngRsltExpr (DmngRsltDecltypeExpr)
    case PSI_Decltype:
      switch (State.DotPos) {
      // <decltype> ::= @ ( Dt | DT ) <expression> E
      case 0:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "Dt")) {
          Ctx.Variables.emplace_back(true);
          State.advance();
          break;
        }
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "DT")) {
          Ctx.Variables.emplace_back(false);
          State.advance();
          break;
        }
        Ctx.reportFailToParent();
        break;
      // <decltype> ::= ( Dt | DT ) @ <expression> E
      case 1:
        Ctx.parseChildState(PSI_Expression);
        break;
      // <decltype> ::= ( Dt | DT ) <expression> @ E
      case 2: {
          if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "E")) {
            Ctx.reportFailToParent();
            break;
          }

          // <decltype> ::= ( Dt | DT ) <expression> E @
          INP_ENSURE_TMP_SIZE(2);

          auto IsSimple = Ctx.getStateVar(0).unsafeGetAsExact<bool>();
          const auto &NdExpr =
            Ctx.getStateVar(1).unsafeGetAsExact<ParseNodeT>();

          auto NdDecltype = std::make_shared<DmngRsltDecltypeExpr>(
            NdExpr->getAs<DNDK_Expr>(), IsSimple);

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdDecltype));

          Ctx.reportSuccessToParent();
        }
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <vendor-qualifier> ::= U <source-name> [<template-args>]
    //
    // Result variables:   DmngRsltVendorQual
    case PSI_VendorQual:
      switch (State.DotPos) {
      // <vendor-qualifier> ::= @ U <source-name> [<template-args>]
      case 0: {
          if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "U")) {
            Ctx.reportFailToParent();
            break;
          }

          // <vendor-qualifier> ::= U @ <source-name> [<template-args>]
          std::string QualName;
          if (!matchSourceNameAndAdvance(Ctx.Result.getMangledName(),
                                          State.Pos, QualName)) {
            Ctx.reportFailToParent();
            break;
          }

          Ctx.Variables.emplace_back(QualName);
          State.addTArgScope();
          State.advance();
        }
        break;
      // <vendor-qualifier> ::= U <source-name> @ [<template-args>]
      case 1:
        Ctx.tryParseChildState(PSI_TemplateArgs, 2, false);
        break;
      // <vendor-qualifier> ::= U <source-name> [<template-args>] @
      case 2: {
          INP_ENSURE_TMP_SIZE(1);

          const auto &QualName =
              Ctx.getStateVar(0).unsafeGetAsExact<std::string>();

          DmngRsltVendorQual Qual(QualName);

          for (std::size_t I = 1, E = Ctx.getStateVarsCount(); I < E; ++I) {
            const auto &Arg =
              Ctx.getStateVar(I).unsafeGetAsExact<DmngRsltTArg>();

              Qual.addTemplateArg(Arg);
          }

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(Qual));

          Ctx.reportSuccessToParent();
        }
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <data-member-prefix> ::= <source-name> M
    //
    // Result variables:   DmngRsltNamePart (DmngRsltSrcNamePart)
    case PSI_DataMemberPrefix:
      switch (State.DotPos) {
      // <data-member-prefix> ::= @ <source-name> M
      case 0: {
          std::string MemberName;
          if (!matchSourceNameAndAdvance(Ctx.Result.getMangledName(),
                                          State.Pos, MemberName)) {
            Ctx.reportFailToParent();
            break;
          }

          // <data-member-prefix> ::= <source-name> @ M
          if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "M")) {
            Ctx.reportFailToParent();
            break;
          }

          // <data-member-prefix> ::= <source-name> M @
          Ctx.Variables.emplace_back(
            std::make_shared<DmngRsltSrcNamePart>(MemberName, true));

          Ctx.reportSuccessToParent();
        }
        break;
      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <unnamed-type-name> ::= Ut [<number>] _
    //                     ::= Ul <type>+ E [<number>] _
    //
    // Result variables:   DmngRsltNamePart (DmngRsltUnmTypeNamePart)
    case PSI_UnnamedTypeName:
      switch (State.DotPos) {
      // <unnamed-type-name> ::= @ Ut [<number>] _
      // Discriminable by prefix -> selecting production in first step.
      case 0: {
          if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "Ut")) {

            // <unnamed-type-name> ::= @ Ul <type>+ E [<number>] _
            if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "Ul")) {
              State.moveToProduction(1, false);
              break;
            }

            Ctx.reportFailToParent();
            break;
          }

          // <unnamed-type-name> ::= Ut @ [<number>] _
          unsigned long long DiscriminatorIdx = 0;
          if (matchIntNumberAndAdvance<unsigned long long, NMK_NonNegative>(
                Ctx.Result.getMangledName(), State.Pos, DiscriminatorIdx)) {
            assert(DiscriminatorIdx + 1 > DiscriminatorIdx &&
                    "Discrimantor index is out of bounds.");
            ++DiscriminatorIdx;
          }

          // <unnamed-type-name> ::= Ut [<number>] @ _
          if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "_")) {
            Ctx.reportFailToParent();
            break;
          }

          // <unnamed-type-name> ::= Ut [<number>] _ @
          Ctx.Variables.emplace_back(
            std::make_shared<DmngRsltUnmTypeNamePart>(DiscriminatorIdx));

          Ctx.reportSuccessToParent();
        }
        break;

      // <unnamed-type-name> ::= Ul @ <type>+ E [<number>] _
      case 1:
        Ctx.parseChildState(PSI_BareFunctionType);
        break;
      // <unnamed-type-name> ::= Ul <type>+ @ E [<number>] _
      case 2: {
          if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "E")) {
            Ctx.reportFailToParent();
            break;
          }

          // <unnamed-type-name> ::= Ul <type>+ E @ [<number>] _
          unsigned long long DiscriminatorIdx = 0;
          if (matchIntNumberAndAdvance<unsigned long long, NMK_NonNegative>(
                Ctx.Result.getMangledName(), State.Pos, DiscriminatorIdx)) {
            assert(DiscriminatorIdx + 1 > DiscriminatorIdx &&
                    "Discrimantor index is out of bounds.");
            ++DiscriminatorIdx;
          }

          // <unnamed-type-name> ::= Ul <type>+ E [<number>] @ _
          if (!matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "_")) {
            Ctx.reportFailToParent();
            break;
          }

          // <unnamed-type-name> ::= Ul <type>+ E [<number>] _ @
          INP_ENSURE_TMP_SIZE(1);

          auto NdClosureType =
            std::make_shared<DmngRsltUnmTypeNamePart>(DiscriminatorIdx);

          for (std::size_t I = 0, E = Ctx.getStateVarsCount(); I < E; ++I) {
            const auto &NdArgType =
              Ctx.getStateVar(I).unsafeGetAsExact<ParseNodeT>();
            NdClosureType->addSignatureType(NdArgType->getAs<DNDK_Type>());
          }

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdClosureType));

          Ctx.reportSuccessToParent();
        }
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <operator-name> ::= <fixed-operator-name>
    //                 ::= cv <type>
    //                 ::= li <source-name>
    //                 ::= v <digit> <source-name>
    //
    // Result variables:   DmngRsltNamePart (DmngRsltOpNamePart)
    case PSI_OperatorName:
      switch (State.DotPos) {
      // <operator-name> ::= @ <fixed-operator-name>
      // Discriminable by prefix -> selecting production in first step.
      case 0: {
          DmngOperatorName OperatorCode = DON_New;
          if (!matchFixedOperatorAndAdvance(Ctx.Result.getMangledName(),
                                            State.Pos, OperatorCode)) {

            // <operator-name> ::= @ cv <type>
            if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "cv")) {
              State.setTArgScope(true);
              State.moveToProduction(1, false);
              break;
            }

            // <operator-name> ::= @ li <source-name>
            if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "li")) {
              State.moveToProduction(3, false);
              break;
            }

            // <operator-name> ::= @ v <digit> <source-name>
            if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                      State.Pos, "v")) {
              State.moveToProduction(4, false);
              break;
            }

            Ctx.reportFailToParent();
            break;
          }

          // <operator-name> ::= <fixed-operator-name> @
          Ctx.Variables.emplace_back(
            std::make_shared<DmngRsltOpNamePart>(OperatorCode));

          Ctx.reportSuccessToParent();
        }
        break;

      // <operator-name> ::= cv @ <type>
      case 1:
        Ctx.parseChildState(PSI_Type);
        break;
      // <operator-name> ::= cv <type> @
      case 2: {
          INP_ENSURE_TMP_SIZE(1);

          auto NdType = Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(
            std::make_shared<DmngRsltOpNamePart>(NdType->getAs<DNDK_Type>()));

          Ctx.reportSuccessToParent();
        }
        break;

      // <operator-name> ::= li @ <source-name>
      case 3: {
          std::string LiteralSuffix;
          if (!matchSourceNameAndAdvance(Ctx.Result.getMangledName(),
                                          State.Pos, LiteralSuffix)) {
            Ctx.reportFailToParent();
            break;
          }

          // <operator-name> ::= li <source-name> @
          Ctx.Variables.emplace_back(
            std::make_shared<DmngRsltOpNamePart>(LiteralSuffix));

          Ctx.reportSuccessToParent();
        }
        break;

      // <operator-name> ::= v @ <digit> <source-name>
      case 4: {
          int Arity = 0;
          if (!matchIntNumberAndAdvance<int, NMK_NonNegative, 1>(
                Ctx.Result.getMangledName(), State.Pos, Arity)) {
            Ctx.reportFailToParent();
            break;
          }

          // <operator-name> ::= v <digit> @ <source-name>
          std::string VendorOpName;
          if (!matchSourceNameAndAdvance(Ctx.Result.getMangledName(),
                                          State.Pos, VendorOpName)) {
            Ctx.reportFailToParent();
            break;
          }

          // <operator-name> ::= v <digit> <source-name> @
          Ctx.Variables.emplace_back(
            std::make_shared<DmngRsltOpNamePart>(VendorOpName, Arity));

          Ctx.reportSuccessToParent();
        }
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <expression> ::= <expr-primary>
    //              ::= <template-param>
    //
    // Result variables:   DmngRsltExpr
    case PSI_Expression:
      switch (State.DotPos) {
      // <expression> ::= @ <expr-primary>
      case 0:
        Ctx.tryParseChildState(PSI_ExprPrimary, 2);
        break;
      // <expression> ::= <expr-primary> @
      case 1:
        Ctx.reportSuccessToParent();
        break;

      // <expression> ::= @ <template-param>
      case 2:
        Ctx.parseChildState(PSI_TemplateParam);
        break;
      // <expression> ::= <template-param> @
      case 3:
        Ctx.reportSuccessToParent();
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    // MC***
    // <expr-primary> ::= L <mangled-name> E
    //                ::= L <type> [<number>] E
    //
    // NOTE: Bug in clang 3.6 with <mangled-name> taken into consideration.
    //
    // Result variables:   DmngRsltExpr (DmngRsltPrimaryExpr)
    case PSI_ExprPrimary:
      switch (State.DotPos) {
      // <expr-primary> ::= @ L <mangled-name> E
      case 0:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "L"))
          State.advance();
        else
          Ctx.reportFailToParent();
        break;
      // <expr-primary> ::= L @ <mangled-name> E
      case 1:
        Ctx.tryParseChildState(PSI_MangledName_Bug, 4, false);
        break;
      // <expr-primary> ::= L <mangled-name> @ E
      case 2:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "E")) {
          // <expr-primary> ::= L <mangled-name> E @
          INP_ENSURE_TMP_SIZE(1);

          const auto &NdExternalName =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>();

          auto NdExternNameExpr = std::make_shared<DmngRsltPrimaryExpr>(
                                    NdExternalName->getAs<DNDK_Name>());

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdExternNameExpr));

          Ctx.reportSuccessToParent();
          break;
        }
        State.moveToProduction(3);
        break;

      // <expr-primary> ::= @ L <type> [<number>] E
      case 3:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "L"))
          State.advance();
        else
          Ctx.reportFailToParent();
        break;
      // <expr-primary> ::= L @ <type> [<number>] E
      case 4:
        Ctx.parseChildState(PSI_Type);
        break;
      // <expr-primary> ::= L <type> @ [<number>] E
      case 5: {
          unsigned long long UIntContent = 0;
          if (matchIntNumberAndAdvance<unsigned long long, NMK_NonNegative>(
                Ctx.Result.getMangledName(), State.Pos, UIntContent)) {
            Ctx.Variables.emplace_back(UIntContent);
          }

          long long SIntContent = 0;
          if (matchIntNumberAndAdvance<long long, NMK_Negative>(
                Ctx.Result.getMangledName(), State.Pos, SIntContent)) {
            Ctx.Variables.emplace_back(SIntContent);
          }

          State.advance();
        }
        break;
      // <expr-primary> ::= L <type> @ [<number>] @ E
      case 6:
        if (matchPrefixAndAdvance(Ctx.Result.getMangledName(),
                                  State.Pos, "E")) {
          // <expr-primary> ::= L <type> [<number>] E @
          INP_ENSURE_TMP_SIZE(1);

          auto &&NdLiteralType =
            Ctx.getStateVar(0).unsafeGetAsExact<ParseNodeT>()
              ->getAs<DNDK_Type>();

          std::shared_ptr<DmngRsltPrimaryExpr> NdLiteralExpr;
          if (Ctx.getStateVarsCount() > 1) {
            if (Ctx.getStateVar(1).isAExact<unsigned long long>()) {
              auto Content =
                Ctx.getStateVar(1).unsafeGetAsExact<unsigned long long>();

              if (NdLiteralType->getKind() == DTK_Builtin &&
                  NdLiteralType->getAs<DTK_Builtin>()
                    ->getBuiltinType() == DBT_Bool) {
                NdLiteralExpr = std::make_shared<DmngRsltPrimaryExpr>(
                                  std::move(NdLiteralType), Content != 0);
              }
              else {
                NdLiteralExpr = std::make_shared<DmngRsltPrimaryExpr>(
                                  std::move(NdLiteralType), Content);
              }
            }
            else if (Ctx.getStateVar(1).isAExact<long long>()) {
              auto Content =
                Ctx.getStateVar(1).unsafeGetAsExact<long long>();
              NdLiteralExpr = std::make_shared<DmngRsltPrimaryExpr>(
                                std::move(NdLiteralType), Content);
            }
            else {
              // ReSharper disable once CppUnreachableCode
              assert(false && "Content type is not supported.");
              NdLiteralExpr = std::make_shared<DmngRsltPrimaryExpr>(
                                std::move(NdLiteralType));
            }
          }
          else {
            NdLiteralExpr = std::make_shared<DmngRsltPrimaryExpr>(
                              std::move(NdLiteralType));
          }

          Ctx.resetStateVars();
          Ctx.Variables.emplace_back(std::move(NdLiteralExpr));

          Ctx.reportSuccessToParent();
          break;
        }
        Ctx.reportFailToParent();
        break;

      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Unexpected state.");
        return Ctx.Result.setFailed();
      }
      break;


    default:
      // ReSharper disable once CppUnreachableCode
      assert(false && "Unexpected state.");
      return Ctx.Result.setFailed();
    }
  }

  // Connect template parameters with arguments.
  if(!Ctx.referenceTParams())
    return Ctx.Result.setFailed();

  return Ctx.Result;

  #undef INP_ENSURE_TMP_SIZE
}
