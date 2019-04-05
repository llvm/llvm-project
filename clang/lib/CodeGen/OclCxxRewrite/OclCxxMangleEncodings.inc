//===- OclCxxMangleEncodings.inc - Encoding tabs of fixed names -*- C++ -*-===//
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


// -----------------------------------------------------------------------------
// BUILT-IN TYPES
// -----------------------------------------------------------------------------

/// \def OCLCXX_MENC_BITYPE(name)
/// Describes built-in type with custom name encoding.
///
/// \param name Token with name of built-in type (used with prefix
///             as enum value).
///
/// Used only to generate enum. Custom parsing must be synchronized separately.
#ifndef OCLCXX_MENC_BITYPE
  #define OCLCXX_MENC_BITYPE(name)
  #define OCLCXX_MENC_BITYPE_DEFAULTED_
#endif

/// \def OCLCXX_MENC_BITYPE_FIXED(name, encName, cxxName)
/// Describes built-in type with fixed name encoding (fixed string to match).
///
/// \param name    Token with name of built-in type (used with prefix
///                as enum value).
/// \param encName String literal which contains name in Itanium encoding.
/// \param cxxName String literal with C++-like name (human readable).
///
/// The information is not used in matchFixedBiTypeAndAdvance function, only
/// to generate enum and corresponding name functions. Please synchronize
/// matcher separately.
#ifndef OCLCXX_MENC_BITYPE_FIXED
  #define OCLCXX_MENC_BITYPE_FIXED(name, encName, cxxName)
  #define OCLCXX_MENC_BITYPE_FIXED_DEFAULTED_
#endif

/// \def OCLCXX_MENC_BITYPE_ALIAS(aliasName, origName)
/// Describes alias name for built-in type.
///
/// \param aliasName Token with alias name of built-in type (used with prefix
///                  as enum value).
/// \param origName  Token with original name referred by alias.
///
/// Used only to generate enum. Custom parsing must be synchronized separately.
#ifndef OCLCXX_MENC_BITYPE_ALIAS
  #define OCLCXX_MENC_BITYPE_ALIAS(aliasName, origName)
  #define OCLCXX_MENC_BITYPE_ALIAS_DEFAULTED_
#endif

// -----------------------------------------------------------------------------

OCLCXX_MENC_BITYPE_FIXED(Void,         "v",  "void")           ///< void
OCLCXX_MENC_BITYPE_FIXED(WChar,        "w",  "wchar_t")        ///< wchar_t
OCLCXX_MENC_BITYPE_FIXED(Bool,         "b",  "bool")           ///< bool
OCLCXX_MENC_BITYPE_FIXED(Char,         "c",  "char")           ///< char
OCLCXX_MENC_BITYPE_FIXED(SChar,        "a",  "signed char")    ///< signed char
OCLCXX_MENC_BITYPE_FIXED(UChar,        "h",  "unsigned char")
                             ///< unsigned char
OCLCXX_MENC_BITYPE_FIXED(Short,        "s",  "short")          ///< short
OCLCXX_MENC_BITYPE_FIXED(UShort,       "t",  "unsigned short")
                             ///< unsigned short
OCLCXX_MENC_BITYPE_FIXED(Int,          "i",  "int")            ///< int
OCLCXX_MENC_BITYPE_FIXED(UInt,         "j",  "unsigned int")   ///< unsigned int
OCLCXX_MENC_BITYPE_FIXED(Long,         "l",  "long")           ///< long
OCLCXX_MENC_BITYPE_FIXED(ULong,        "m",  "unsigned long")
                             ///< unsigned long
OCLCXX_MENC_BITYPE_FIXED(LongLong,     "x",  "long long")
                             ///< long long, __int64
OCLCXX_MENC_BITYPE_ALIAS(Int64,        LongLong)
OCLCXX_MENC_BITYPE_FIXED(ULongLong,    "y",  "unsigned long long")
                             ///< unsigned long long, __int64
OCLCXX_MENC_BITYPE_ALIAS(UInt64,       ULongLong)
OCLCXX_MENC_BITYPE_FIXED(Int128,       "n",  "__int128")       ///< __int128
OCLCXX_MENC_BITYPE_FIXED(UInt128,      "o",  "unsigned __int128")
                             ///< unsigned __int128
OCLCXX_MENC_BITYPE_FIXED(Float,        "f",  "float")          ///< float
OCLCXX_MENC_BITYPE_FIXED(Double,       "d",  "double")         ///< double
OCLCXX_MENC_BITYPE_FIXED(LongDouble,   "e",  "long double")
                             ///< long double, __float80
OCLCXX_MENC_BITYPE_ALIAS(Float80,      LongDouble)
OCLCXX_MENC_BITYPE_FIXED(Float128,     "g",  "__float128")     ///< __float128
OCLCXX_MENC_BITYPE_FIXED(Ellipsis,     "z",  "...")            ///< ellipsis
OCLCXX_MENC_BITYPE_FIXED(Float64R,     "Dd", "__fp64r")
                             ///< IEEE 754r decimal floating point (64 bits)
OCLCXX_MENC_BITYPE_FIXED(Float128R,    "De", "__fp128r")
                             ///< IEEE 754r decimal floating point (128 bits)
OCLCXX_MENC_BITYPE_FIXED(Float32R,     "Df", "__fp32r")
                             ///< IEEE 754r decimal floating point (32 bits)
OCLCXX_MENC_BITYPE_FIXED(Float16R,     "DF16_", "_Float16")
                             ///< IEEE 754r half-prec floating point (16 bits)
OCLCXX_MENC_BITYPE_ALIAS(Half,         Float16R)
OCLCXX_MENC_BITYPE_FIXED(Char32,       "Di", "char32_t")       ///< char32_t
OCLCXX_MENC_BITYPE_FIXED(Char16,       "Ds", "char16_t")       ///< char16_t
OCLCXX_MENC_BITYPE_FIXED(Auto,         "Da", "auto")           ///< auto
OCLCXX_MENC_BITYPE_FIXED(DecltypeAuto, "Dc", "decltype(auto)")
                             ///< decltype(auto)
OCLCXX_MENC_BITYPE_FIXED(NullPtr,      "Dn", "decltype(nullptr)")
                             ///< std::nullptr_t (i.e., decltype(nullptr))
OCLCXX_MENC_BITYPE_FIXED(Pixel,        "p",  "pixel")
                             ///< pixel (only for altivec vectors).

OCLCXX_MENC_BITYPE(Vendor) ///< vendor-extended (`source-name`)

// -----------------------------------------------------------------------------

#ifdef OCLCXX_MENC_BITYPE_DEFAULTED_
  #undef OCLCXX_MENC_BITYPE
  #undef OCLCXX_MENC_BITYPE_DEFAULTED_
#endif

#ifdef OCLCXX_MENC_BITYPE_FIXED_DEFAULTED_
  #undef OCLCXX_MENC_BITYPE_FIXED
  #undef OCLCXX_MENC_BITYPE_FIXED_DEFAULTED_
#endif

#ifdef OCLCXX_MENC_BITYPE_ALIAS_DEFAULTED_
  #undef OCLCXX_MENC_BITYPE_ALIAS
  #undef OCLCXX_MENC_BITYPE_ALIAS_DEFAULTED_
#endif


// -----------------------------------------------------------------------------
// OPERATORS
// -----------------------------------------------------------------------------


/// \def OCLCXX_MENC_OPR(name, arity)
/// Describes operator with custom name encoding.
///
/// \param name  Token with name of operator (used with prefix
///              as enum value).
/// \param arity Integral literal with operator arity in <expression> context.
///              Specify 0 if arity is not fixed (variable, unknown or special
///              treatment).
///
/// Used only to generate enum and arity functions. Custom parsing must be
/// synchronized separately.
#ifndef OCLCXX_MENC_OPR
  #define OCLCXX_MENC_OPR(name, arity)
  #define OCLCXX_MENC_OPR_DEFAULTED_
#endif

/// \def OCLCXX_MENC_OPR_FIXED(name, encName, arity, cxxName)
/// Describes operator with fixed name encoding (fixed string to match).
///
/// \param name    Token with name of operator (used with prefix
///                as enum value).
/// \param encName String literal which contains name in Itanium encoding.
/// \param arity   Integral literal with operator arity in <expression> context.
///                Specify 0 if arity is not fixed (variable, unknown or special
///                treatment).
/// \param cxxName String literal with C++-like name (human readable without
///                "operator" prefix).
///
/// The information is not used in matchFixedOperatorAndAdvance function, only
/// to generate enum and corresponding name/arity functions. Please synchronize
/// matcher separately.
#ifndef OCLCXX_MENC_OPR_FIXED
  #define OCLCXX_MENC_OPR_FIXED(name, encName, arity, cxxName)
  #define OCLCXX_MENC_OPR_FIXED_DEFAULTED_
#endif

/// \def OCLCXX_MENC_OPR_ALIAS(aliasName, origName)
/// Describes alias name for built-in type.
///
/// \param aliasName Token with alias name of operator (used with prefix
///                  as enum value).
/// \param origName  Token with original name referred by alias.
///
/// Used only to generate enum. Custom parsing must be synchronized separately.
#ifndef OCLCXX_MENC_OPR_ALIAS
  #define OCLCXX_MENC_OPR_ALIAS(aliasName, origName)
  #define OCLCXX_MENC_OPR_ALIAS_DEFAULTED_
#endif

// -----------------------------------------------------------------------------

/// | operator new      | (new (E1 ...) E)     | special, variable arity |
OCLCXX_MENC_OPR_FIXED(New,                      "nw", 0, "new")
/// | operator new[]    | (new (E2 ...) E[E1]) | special, variable arity |
OCLCXX_MENC_OPR_FIXED(NewArray,                 "na", 0, "new[]")
/// | operator delete   | delete E             | special, unary in expr. |
OCLCXX_MENC_OPR_FIXED(Delete,                   "dl", 0, "delete")
/// | operator delete[] | delete E[]           | special, unary in expr. |
OCLCXX_MENC_OPR_FIXED(DeleteArray,              "da", 0, "delete[]")
/// | operator +        | (+E)                 | unary in expr.          |
OCLCXX_MENC_OPR_FIXED(Positive,                 "ps", 1, "+")
/// | operator -        | (-E)                 | unary in expr.          |
OCLCXX_MENC_OPR_FIXED(Negative,                 "ng", 1, "-")
/// | operator &        | (&E)                 | unary in expr.          |
OCLCXX_MENC_OPR_FIXED(AddressOf,                "ad", 1, "&")
// Indirection operator.
/// | operator *        | (*E)                 | unary in expr.          |
OCLCXX_MENC_OPR_FIXED(Dereference,              "de", 1, "*")
/// | operator ~        | (~E)                 |                         |
OCLCXX_MENC_OPR_FIXED(BitwiseComplement,        "co", 1, "~")
/// | operator +        | (E + E1)             |                         |
OCLCXX_MENC_OPR_FIXED(Plus,                     "pl", 2, "+")
/// | operator -        | (E - E1)             |                         |
OCLCXX_MENC_OPR_FIXED(Minus,                    "mi", 2, "-")
/// | operator *        | (E * E1)             |                         |
OCLCXX_MENC_OPR_FIXED(Multiply,                 "ml", 2, "*")
/// | operator /        | (E / E1)             |                         |
OCLCXX_MENC_OPR_FIXED(Divide,                   "dv", 2, "/")
// Modulus operator.
/// | operator %        | (E % E1)             |                         |
OCLCXX_MENC_OPR_FIXED(Remainder,                "rm", 2, "%")
/// | operator &        | (E & E1)             |                         |
OCLCXX_MENC_OPR_FIXED(BitwiseAnd,               "an", 2, "&")
/// | operator |        | (E | E1)             |                         |
OCLCXX_MENC_OPR_FIXED(BitwiseOr,                "or", 2, "|")
/// | operator ^        | (E ^ E1)             |                         |
OCLCXX_MENC_OPR_FIXED(BitwiseExclusiveOr,       "eo", 2, "^")
/// | operator =        | (E = E1)             |                         |
OCLCXX_MENC_OPR_FIXED(Assign,                   "aS", 2, "=")
/// | operator +=       | (E += E1)            |                         |
OCLCXX_MENC_OPR_FIXED(PlusAssign,               "pL", 2, "+=")
/// | operator -=       | (E -= E1)            |                         |
OCLCXX_MENC_OPR_FIXED(MinusAssign,              "mI", 2, "-=")
/// | operator *=       | (E *= E1)            |                         |
OCLCXX_MENC_OPR_FIXED(MultiplyAssign,           "mL", 2, "*=")
/// | operator /=       | (E /= E1)            |                         |
OCLCXX_MENC_OPR_FIXED(DivideAssign,             "dV", 2, "/=")
/// | operator %=       | (E %= E1)            |                         |
OCLCXX_MENC_OPR_FIXED(RemainderAssign,          "rM", 2, "%=")
/// | operator &=       | (E &= E1)            |                         |
OCLCXX_MENC_OPR_FIXED(BitwiseAndAssign,         "aN", 2, "&=")
/// | operator |=       | (E |= E1)            |                         |
OCLCXX_MENC_OPR_FIXED(BitwiseOrAssign,          "oR", 2, "|=")
/// | operator ^=       | (E ^= E1)            |                         |
OCLCXX_MENC_OPR_FIXED(BitwiseExclusiveOrAssign, "eO", 2, "^=")
/// | operator <<       | (E << E1)            |                         |
OCLCXX_MENC_OPR_FIXED(LeftShift,                "ls", 2, "<<")
/// | operator >>       | (E >> E1)            |                         |
OCLCXX_MENC_OPR_FIXED(RightShift,               "rs", 2, ">>")
/// | operator <<=      | (E <<= E1)           |                         |
OCLCXX_MENC_OPR_FIXED(LeftShiftAssign,          "lS", 2, "<<=")
/// | operator >>=      | (E >>= E1)           |                         |
OCLCXX_MENC_OPR_FIXED(RightShiftAssign,         "rS", 2, ">>=")
/// | operator ==       | (E == E1)            |                         |
OCLCXX_MENC_OPR_FIXED(Equal,                    "eq", 2, "==")
/// | operator !=       | (E != E1)            |                         |
OCLCXX_MENC_OPR_FIXED(NotEqual,                 "ne", 2, "!=")
/// | operator <        | (E < E1)             |                         |
OCLCXX_MENC_OPR_FIXED(LessThan,                 "lt", 2, "<")
/// | operator >        | (E > E1)             |                         |
OCLCXX_MENC_OPR_FIXED(GreaterThan,              "gt", 2, ">")
/// | operator <=       | (E <= E1)            |                         |
OCLCXX_MENC_OPR_FIXED(LessEqual,                "le", 2, "<=")
/// | operator >=       | (E >= E1)            |                         |
OCLCXX_MENC_OPR_FIXED(GreaterEqual,             "ge", 2, ">=")
/// | operator !        | (!E)                 |                         |
OCLCXX_MENC_OPR_FIXED(LogicalNegate,            "nt", 1, "!")
/// | operator &&       | (E && E1)            |                         |
OCLCXX_MENC_OPR_FIXED(LogicalAnd,               "aa", 2, "&&")
/// | operator ||       | (E || E1)            |                         |
OCLCXX_MENC_OPR_FIXED(LogicalOr,                "oo", 2, "||")
/// | operator ++       | (E++) or (++E)       | postfix in expr.        |
OCLCXX_MENC_OPR_FIXED(PlusPlus,                 "pp", 1, "++")
/// | operator --       | (E--) or (--E)       | postfix in expr.        |
OCLCXX_MENC_OPR_FIXED(MinusMinus,               "mm", 1, "--")
/// | operator ,        | (E, E1)              |                         |
OCLCXX_MENC_OPR_FIXED(Comma,                    "cm", 2, ",")
/// | operator ->*      | (E->*E1)             |                         |
OCLCXX_MENC_OPR_FIXED(PointerToMemberAccess,    "pm", 2, "->*")
// Arrow operator ("->"). Special treatment required because second operand
// is not an expression, but name.
/// | operator ->       | (E->E1)              | special, binary         |
OCLCXX_MENC_OPR_FIXED(MemberAccess,             "pt", 0, "->")
/// | operator ()       | (E(E1 ...))          | variable arity          |
OCLCXX_MENC_OPR_FIXED(Call,                     "cl", 0, "()")
  // Subscript operator.
/// | operator []       | (E[E1])              |                         |
OCLCXX_MENC_OPR_FIXED(Index,                    "ix", 2, "[]")
  // Ternary operator.
/// | operator ?        | (E ? E1 : E2)        |                         |
OCLCXX_MENC_OPR_FIXED(Conditional,              "qu", 3, "?")

// Operators with names of variable size.
/// | operator <type>    | (<type>(E ...)) or ((<type>)E) | variable arity     |
OCLCXX_MENC_OPR(Convert, 0)
/// | operator "" <name> | (E<name>)                      |                    |
OCLCXX_MENC_OPR(Literal, 1)
/// | operator <name>    | (operator `<name>`(E ...))     | arity from parsing |
OCLCXX_MENC_OPR(Vendor,  0)

// -----------------------------------------------------------------------------

#ifdef OCLCXX_MENC_OPR_DEFAULTED_
  #undef OCLCXX_MENC_OPR
  #undef OCLCXX_MENC_OPR_DEFAULTED_
#endif

#ifdef OCLCXX_MENC_OPR_FIXED_DEFAULTED_
  #undef OCLCXX_MENC_OPR_FIXED
  #undef OCLCXX_MENC_OPR_FIXED_DEFAULTED_
#endif

#ifdef OCLCXX_MENC_OPR_ALIAS_DEFAULTED_
  #undef OCLCXX_MENC_OPR_ALIAS
  #undef OCLCXX_MENC_OPR_ALIAS_DEFAULTED_
#endif
