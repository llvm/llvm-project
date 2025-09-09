//===--- ItaniumRTTIBuilder.cpp - LLVM Backend Utilities --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGenShared/ItaniumRTTIBuilder.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"

namespace clang::CodeGenShared {

/// TypeInfoIsInStandardLibrary - Given a builtin type, returns whether the type
/// info for that type is defined in the standard library.
bool TypeInfoIsInStandardLibrary(const BuiltinType *Ty) {
  // Itanium C++ ABI 2.9.2:
  //   Basic type information (e.g. for "int", "bool", etc.) will be kept in
  //   the run-time support library. Specifically, the run-time support
  //   library should contain type_info objects for the types X, X* and
  //   X const*, for every X in: void, std::nullptr_t, bool, wchar_t, char,
  //   unsigned char, signed char, short, unsigned short, int, unsigned int,
  //   long, unsigned long, long long, unsigned long long, float, double,
  //   long double, char16_t, char32_t, and the IEEE 754r decimal and
  //   half-precision floating point types.
  //
  // GCC also emits RTTI for __int128.
  // FIXME: We do not emit RTTI information for decimal types here.

  // Types added here must also be added to EmitFundamentalRTTIDescriptors.
  switch (Ty->getKind()) {
  case BuiltinType::Void:
  case BuiltinType::NullPtr:
  case BuiltinType::Bool:
  case BuiltinType::WChar_S:
  case BuiltinType::WChar_U:
  case BuiltinType::Char_U:
  case BuiltinType::Char_S:
  case BuiltinType::UChar:
  case BuiltinType::SChar:
  case BuiltinType::Short:
  case BuiltinType::UShort:
  case BuiltinType::Int:
  case BuiltinType::UInt:
  case BuiltinType::Long:
  case BuiltinType::ULong:
  case BuiltinType::LongLong:
  case BuiltinType::ULongLong:
  case BuiltinType::Half:
  case BuiltinType::Float:
  case BuiltinType::Double:
  case BuiltinType::LongDouble:
  case BuiltinType::Float16:
  case BuiltinType::Float128:
  case BuiltinType::Ibm128:
  case BuiltinType::Char8:
  case BuiltinType::Char16:
  case BuiltinType::Char32:
  case BuiltinType::Int128:
  case BuiltinType::UInt128:
    return true;

#define IMAGE_TYPE(ImgType, Id, SingletonId, Access, Suffix)                   \
  case BuiltinType::Id:
#include "clang/Basic/OpenCLImageTypes.def"
#define EXT_OPAQUE_TYPE(ExtType, Id, Ext) case BuiltinType::Id:
#include "clang/Basic/OpenCLExtensionTypes.def"
  case BuiltinType::OCLSampler:
  case BuiltinType::OCLEvent:
  case BuiltinType::OCLClkEvent:
  case BuiltinType::OCLQueue:
  case BuiltinType::OCLReserveID:
#define SVE_TYPE(Name, Id, SingletonId) case BuiltinType::Id:
#include "clang/Basic/AArch64ACLETypes.def"
#define PPC_VECTOR_TYPE(Name, Id, Size) case BuiltinType::Id:
#include "clang/Basic/PPCTypes.def"
#define RVV_TYPE(Name, Id, SingletonId) case BuiltinType::Id:
#include "clang/Basic/RISCVVTypes.def"
#define WASM_TYPE(Name, Id, SingletonId) case BuiltinType::Id:
#include "clang/Basic/WebAssemblyReferenceTypes.def"
#define AMDGPU_TYPE(Name, Id, SingletonId, Width, Align) case BuiltinType::Id:
#include "clang/Basic/AMDGPUTypes.def"
#define HLSL_INTANGIBLE_TYPE(Name, Id, SingletonId) case BuiltinType::Id:
#include "clang/Basic/HLSLIntangibleTypes.def"
  case BuiltinType::ShortAccum:
  case BuiltinType::Accum:
  case BuiltinType::LongAccum:
  case BuiltinType::UShortAccum:
  case BuiltinType::UAccum:
  case BuiltinType::ULongAccum:
  case BuiltinType::ShortFract:
  case BuiltinType::Fract:
  case BuiltinType::LongFract:
  case BuiltinType::UShortFract:
  case BuiltinType::UFract:
  case BuiltinType::ULongFract:
  case BuiltinType::SatShortAccum:
  case BuiltinType::SatAccum:
  case BuiltinType::SatLongAccum:
  case BuiltinType::SatUShortAccum:
  case BuiltinType::SatUAccum:
  case BuiltinType::SatULongAccum:
  case BuiltinType::SatShortFract:
  case BuiltinType::SatFract:
  case BuiltinType::SatLongFract:
  case BuiltinType::SatUShortFract:
  case BuiltinType::SatUFract:
  case BuiltinType::SatULongFract:
  case BuiltinType::BFloat16:
    return false;

  case BuiltinType::Dependent:
#define BUILTIN_TYPE(Id, SingletonId)
#define PLACEHOLDER_TYPE(Id, SingletonId) case BuiltinType::Id:
#include "clang/AST/BuiltinTypes.def"
    llvm_unreachable("asking for RRTI for a placeholder type!");

  case BuiltinType::ObjCId:
  case BuiltinType::ObjCClass:
  case BuiltinType::ObjCSel:
    llvm_unreachable("FIXME: Objective-C types are unsupported!");
  }

  llvm_unreachable("Invalid BuiltinType Kind!");
}

bool TypeInfoIsInStandardLibrary(const PointerType *PointerTy) {
  QualType PointeeTy = PointerTy->getPointeeType();
  const BuiltinType *BuiltinTy = dyn_cast<BuiltinType>(PointeeTy);
  if (!BuiltinTy)
    return false;

  // Check the qualifiers.
  Qualifiers Quals = PointeeTy.getQualifiers();
  Quals.removeConst();

  if (!Quals.empty())
    return false;

  return TypeInfoIsInStandardLibrary(BuiltinTy);
}

/// IsStandardLibraryRTTIDescriptor - Returns whether the type
/// information for the given type exists in the standard library.
bool IsStandardLibraryRTTIDescriptor(QualType Ty) {
  // Type info for builtin types is defined in the standard library.
  if (const BuiltinType *BuiltinTy = dyn_cast<BuiltinType>(Ty))
    return TypeInfoIsInStandardLibrary(BuiltinTy);

  // Type info for some pointer types to builtin types is defined in the
  // standard library.
  if (const PointerType *PointerTy = dyn_cast<PointerType>(Ty))
    return TypeInfoIsInStandardLibrary(PointerTy);

  return false;
}

/// IsIncompleteClassType - Returns whether the given record type is incomplete.
bool IsIncompleteClassType(const RecordType *RecordTy) {
  return !RecordTy->getOriginalDecl()
              ->getDefinitionOrSelf()
              ->isCompleteDefinition();
}

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
bool ContainsIncompleteClassType(QualType Ty) {
  if (const RecordType *RecordTy = dyn_cast<RecordType>(Ty)) {
    if (IsIncompleteClassType(RecordTy))
      return true;
  }

  if (const PointerType *PointerTy = dyn_cast<PointerType>(Ty))
    return ContainsIncompleteClassType(PointerTy->getPointeeType());

  if (const MemberPointerType *MemberPointerTy =
          dyn_cast<MemberPointerType>(Ty)) {
    // Check if the class type is incomplete.
    if (!MemberPointerTy->getMostRecentCXXRecordDecl()->hasDefinition())
      return true;

    return ContainsIncompleteClassType(MemberPointerTy->getPointeeType());
  }

  return false;
}

// CanUseSingleInheritance - Return whether the given record decl has a "single,
// public, non-virtual base at offset zero (i.e. the derived class is dynamic
// iff the base is)", according to Itanium C++ ABI, 2.95p6b.
bool CanUseSingleInheritance(const CXXRecordDecl *RD) {
  // Check the number of bases.
  if (RD->getNumBases() != 1)
    return false;

  // Get the base.
  CXXRecordDecl::base_class_const_iterator Base = RD->bases_begin();

  // Check that the base is not virtual.
  if (Base->isVirtual())
    return false;

  // Check that the base is public.
  if (Base->getAccessSpecifier() != AS_public)
    return false;

  // Check that the class is dynamic iff the base is.
  auto *BaseDecl = Base->getType()->castAsCXXRecordDecl();
  if (!BaseDecl->isEmpty() &&
      BaseDecl->isDynamicClass() != RD->isDynamicClass())
    return false;

  return true;
}

const char *VTableClassNameForType(const Type *Ty) {
  // abi::__class_type_info.
  static const char *const ClassTypeInfo =
      "_ZTVN10__cxxabiv117__class_type_infoE";
  // abi::__si_class_type_info.
  static const char *const SIClassTypeInfo =
      "_ZTVN10__cxxabiv120__si_class_type_infoE";
  // abi::__vmi_class_type_info.
  static const char *const VMIClassTypeInfo =
      "_ZTVN10__cxxabiv121__vmi_class_type_infoE";

  switch (Ty->getTypeClass()) {
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.inc"
    llvm_unreachable("Non-canonical and dependent types shouldn't get here");

  case Type::LValueReference:
  case Type::RValueReference:
    llvm_unreachable("References shouldn't get here");

  case Type::Auto:
  case Type::DeducedTemplateSpecialization:
    llvm_unreachable("Undeduced type shouldn't get here");

  case Type::Pipe:
    llvm_unreachable("Pipe types shouldn't get here");

  case Type::ArrayParameter:
    llvm_unreachable("Array Parameter types should not get here.");

  case Type::Builtin:
  case Type::BitInt:
  // GCC treats vector and complex types as fundamental types.
  case Type::Vector:
  case Type::ExtVector:
  case Type::ConstantMatrix:
  case Type::Complex:
  case Type::Atomic:
  // FIXME: GCC treats block pointers as fundamental types?!
  case Type::BlockPointer:
    // abi::__fundamental_type_info.
    return "_ZTVN10__cxxabiv123__fundamental_type_infoE";

  case Type::ConstantArray:
  case Type::IncompleteArray:
  case Type::VariableArray:
    // abi::__array_type_info.
    return "_ZTVN10__cxxabiv117__array_type_infoE";

  case Type::FunctionNoProto:
  case Type::FunctionProto:
    // abi::__function_type_info.
    return "_ZTVN10__cxxabiv120__function_type_infoE";

  case Type::Enum:
    // abi::__enum_type_info.
    return "_ZTVN10__cxxabiv116__enum_type_infoE";

  case Type::Record: {
    const CXXRecordDecl *RD =
        cast<CXXRecordDecl>(cast<RecordType>(Ty)->getOriginalDecl())
            ->getDefinitionOrSelf();

    if (!RD->hasDefinition() || !RD->getNumBases()) {
      return ClassTypeInfo;
    }

    if (CanUseSingleInheritance(RD)) {
      return SIClassTypeInfo;
    }

    return VMIClassTypeInfo;
  }

  case Type::ObjCObject:
    // Ignore protocol qualifiers.
    Ty = cast<ObjCObjectType>(Ty)->getBaseType().getTypePtr();

    // Handle id and Class.
    if (isa<BuiltinType>(Ty)) {
      return ClassTypeInfo;
    }

    assert(isa<ObjCInterfaceType>(Ty));
    [[fallthrough]];

  case Type::ObjCInterface:
    if (cast<ObjCInterfaceType>(Ty)->getDecl()->getSuperClass()) {
      return SIClassTypeInfo;
    } else {
      return ClassTypeInfo;
    }

  case Type::ObjCObjectPointer:
  case Type::Pointer:
    // abi::__pointer_type_info.
    return "_ZTVN10__cxxabiv119__pointer_type_infoE";

  case Type::MemberPointer:
    // abi::__pointer_to_member_type_info.
    return "_ZTVN10__cxxabiv129__pointer_to_member_type_infoE";

  case Type::HLSLAttributedResource:
  case Type::HLSLInlineSpirv:
    llvm_unreachable("HLSL doesn't support virtual functions");
  }

  return nullptr;
}

/// Compute the flags for a __pbase_type_info, and remove the corresponding
/// pieces from \p Type.
unsigned extractPBaseFlags(ASTContext &Ctx, QualType &Type) {
  unsigned Flags = 0;

  if (Type.isConstQualified())
    Flags |= PTI_Const;
  if (Type.isVolatileQualified())
    Flags |= PTI_Volatile;
  if (Type.isRestrictQualified())
    Flags |= PTI_Restrict;
  Type = Type.getUnqualifiedType();

  // Itanium C++ ABI 2.9.5p7:
  //   When the abi::__pbase_type_info is for a direct or indirect pointer to an
  //   incomplete class type, the incomplete target type flag is set.
  if (ContainsIncompleteClassType(Type))
    Flags |= PTI_Incomplete;

  if (auto *Proto = Type->getAs<FunctionProtoType>()) {
    if (Proto->isNothrow()) {
      Flags |= PTI_Noexcept;
      Type = Ctx.getFunctionTypeWithExceptionSpec(Type, EST_None);
    }
  }

  return Flags;
}

/// ComputeVMIClassTypeInfoFlags - Compute the value of the flags member in
/// abi::__vmi_class_type_info.
///
unsigned ComputeVMIClassTypeInfoFlags(const CXXBaseSpecifier *Base,
                                      SeenBases &Bases) {

  unsigned Flags = 0;

  auto *BaseDecl = Base->getType()->castAsCXXRecordDecl();
  if (Base->isVirtual()) {
    // Mark the virtual base as seen.
    if (!Bases.VirtualBases.insert(BaseDecl).second) {
      // If this virtual base has been seen before, then the class is diamond
      // shaped.
      Flags |= VMI_DiamondShaped;
    } else {
      if (Bases.NonVirtualBases.count(BaseDecl))
        Flags |= VMI_NonDiamondRepeat;
    }
  } else {
    // Mark the non-virtual base as seen.
    if (!Bases.NonVirtualBases.insert(BaseDecl).second) {
      // If this non-virtual base has been seen before, then the class has non-
      // diamond shaped repeated inheritance.
      Flags |= VMI_NonDiamondRepeat;
    } else {
      if (Bases.VirtualBases.count(BaseDecl))
        Flags |= VMI_NonDiamondRepeat;
    }
  }

  // Walk all bases.
  for (const auto &I : BaseDecl->bases())
    Flags |= ComputeVMIClassTypeInfoFlags(&I, Bases);

  return Flags;
}

unsigned ComputeVMIClassTypeInfoFlags(const CXXRecordDecl *RD) {
  unsigned Flags = 0;
  SeenBases Bases;

  // Walk all bases.
  for (const auto &I : RD->bases())
    Flags |= ComputeVMIClassTypeInfoFlags(&I, Bases);

  return Flags;
}
} // namespace clang::CodeGenShared
