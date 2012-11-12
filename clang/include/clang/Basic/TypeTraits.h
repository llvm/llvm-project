//===--- TypeTraits.h - C++ Type Traits Support Enumerations ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines enumerations for the type traits support.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TYPETRAITS_H
#define LLVM_CLANG_TYPETRAITS_H

namespace clang {

  /// \brief Names for the unary type traits.
  enum UnaryTypeTrait {
    UTT_HasNothrowAssign,
    UTT_HasNothrowCopy,
    UTT_HasNothrowConstructor,
    UTT_HasTrivialAssign,
    UTT_HasTrivialCopy,
    UTT_HasTrivialDefaultConstructor,
    UTT_HasTrivialDestructor,
    UTT_HasVirtualDestructor,
    UTT_IsAbstract,
    UTT_IsArithmetic,
    UTT_IsArray,
    UTT_IsClass,
    UTT_IsCompleteType,
    UTT_IsCompound,
    UTT_IsConst,
    UTT_IsEmpty,
    UTT_IsEnum,
    UTT_IsFinal,
    UTT_IsFloatingPoint,
    UTT_IsFunction,
    UTT_IsFundamental,
    UTT_IsIntegral,
    UTT_IsInterfaceClass,
    UTT_IsLiteral,
    UTT_IsLvalueReference,
    UTT_IsMemberFunctionPointer,
    UTT_IsMemberObjectPointer,
    UTT_IsMemberPointer,
    UTT_IsObject,
    UTT_IsPOD,
    UTT_IsPointer,
    UTT_IsPolymorphic,
    UTT_IsReference,
    UTT_IsRvalueReference,
    UTT_IsScalar,
    UTT_IsSigned,
    UTT_IsStandardLayout,
    UTT_IsTrivial,
    UTT_IsTriviallyCopyable,
    UTT_IsUnion,
    UTT_IsUnsigned,
    UTT_IsVoid,
    UTT_IsVolatile
  };

  /// \brief Names for the binary type traits.
  enum BinaryTypeTrait {
    BTT_IsBaseOf,
    BTT_IsConvertible,
    BTT_IsConvertibleTo,
    BTT_IsSame,
    BTT_TypeCompatible,
    BTT_IsTriviallyAssignable
  };

  /// \brief Names for the array type traits.
  enum ArrayTypeTrait {
    ATT_ArrayRank,
    ATT_ArrayExtent
  };

  /// \brief Names for the "expression or type" traits.
  enum UnaryExprOrTypeTrait {
    UETT_SizeOf,
    UETT_AlignOf,
    UETT_VecStep
  };
  
  /// \brief Names for type traits that operate specifically on types.
  enum TypeTrait {
    TT_IsTriviallyConstructible
  };
  
}

#endif
