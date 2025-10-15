//===--- ItaniumRTTIBuilder.h - LLVM Backend Utilities ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGENSHARED_ITANIUMRTTIBUILDER_H
#define LLVM_CLANG_LIB_CODEGENSHARED_ITANIUMRTTIBUILDER_H

#include "clang/AST/DeclCXX.h"
#include "clang/AST/TypeBase.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace clang {

namespace CodeGenShared {

// Pointer type info flags.
enum {
  /// PTI_Const - Type has const qualifier.
  PTI_Const = 0x1,

  /// PTI_Volatile - Type has volatile qualifier.
  PTI_Volatile = 0x2,

  /// PTI_Restrict - Type has restrict qualifier.
  PTI_Restrict = 0x4,

  /// PTI_Incomplete - Type is incomplete.
  PTI_Incomplete = 0x8,

  /// PTI_ContainingClassIncomplete - Containing class is incomplete.
  /// (in pointer to member).
  PTI_ContainingClassIncomplete = 0x10,

  /// PTI_TransactionSafe - Pointee is transaction_safe function (C++ TM TS).
  // PTI_TransactionSafe = 0x20,

  /// PTI_Noexcept - Pointee is noexcept function (C++1z).
  PTI_Noexcept = 0x40,
};

// VMI type info flags.
enum {
  /// VMI_NonDiamondRepeat - Class has non-diamond repeated inheritance.
  VMI_NonDiamondRepeat = 0x1,

  /// VMI_DiamondShaped - Class is diamond shaped.
  VMI_DiamondShaped = 0x2
};

// Base class type info flags.
enum {
  /// BCTI_Virtual - Base class is virtual.
  BCTI_Virtual = 0x1,

  /// BCTI_Public - Base class is public.
  BCTI_Public = 0x2
};

/// TypeInfoIsInStandardLibrary - Given a builtin type, returns whether the type
/// info for that type is defined in the standard library.
bool TypeInfoIsInStandardLibrary(const BuiltinType *Ty);

bool TypeInfoIsInStandardLibrary(const PointerType *PointerTy);

/// IsStandardLibraryRTTIDescriptor - Returns whether the type
/// information for the given type exists in the standard library.
bool IsStandardLibraryRTTIDescriptor(QualType Ty);

/// IsIncompleteClassType - Returns whether the given record type is incomplete.
bool IsIncompleteClassType(const RecordType *RecordTy);

/// ContainsIncompleteClassType - Returns whether the given type contains an
/// incomplete class type. This is true if
///
///   * The given type is an incomplete class type.
///   * The given type is a pointer type whose pointee type contains an
///     incomplete class type.
///   * The given type is a member pointer type whose class is an incomplete
///     class type.
///   * The given type is a member pointer type whoise pointee type contains an
///     incomplete class type.
/// is an indirect or direct pointer to an incomplete class type.
bool ContainsIncompleteClassType(QualType Ty);

// CanUseSingleInheritance - Return whether the given record decl has a "single,
// public, non-virtual base at offset zero (i.e. the derived class is dynamic
// iff the base is)", according to Itanium C++ ABI, 2.95p6b.
bool CanUseSingleInheritance(const CXXRecordDecl *RD);

const char *VTableClassNameForType(const Type *Ty);

/// Compute the flags for a __pbase_type_info, and remove the corresponding
/// pieces from \p Type.
unsigned extractPBaseFlags(ASTContext &Ctx, QualType &Type);

/// SeenBases - Contains virtual and non-virtual bases seen when traversing
/// a class hierarchy.
struct SeenBases {
  llvm::SmallPtrSet<const CXXRecordDecl *, 16> NonVirtualBases;
  llvm::SmallPtrSet<const CXXRecordDecl *, 16> VirtualBases;
};

/// ComputeVMIClassTypeInfoFlags - Compute the value of the flags member in
/// abi::__vmi_class_type_info.
unsigned ComputeVMIClassTypeInfoFlags(const CXXBaseSpecifier *Base,
                                      SeenBases &Bases);

unsigned ComputeVMIClassTypeInfoFlags(const CXXRecordDecl *RD);

} // namespace CodeGenShared
} // namespace clang

#endif
