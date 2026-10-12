//===--- ItaniumCXXABIUtils.cpp - Shared Itanium C++ ABI queries ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGenUtils/ItaniumCXXABIUtils.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/RecordLayout.h"

namespace clang::CodeGenUtils {

CharUnits computeOffsetHint(ASTContext &Ctx, const CXXRecordDecl *Src,
                            const CXXRecordDecl *Dst) {
  CXXBasePaths Paths(/*FindAmbiguities=*/true, /*RecordPaths=*/true,
                     /*DetectVirtual=*/false);

  // If Dst is not derived from Src we can skip the whole computation below and
  // return that Src is not a public base of Dst.  Record all inheritance paths.
  if (!Dst->isDerivedFrom(Src, Paths))
    return CharUnits::fromQuantity(-2ULL);

  unsigned NumPublicPaths = 0;
  CharUnits Offset;

  // Now walk all possible inheritance paths.
  for (const CXXBasePath &Path : Paths) {
    if (Path.Access != AS_public) // Ignore non-public inheritance.
      continue;

    ++NumPublicPaths;

    for (const CXXBasePathElement &PathElement : Path) {
      // If the path contains a virtual base class we can't give any hint.
      // -1: no hint.
      if (PathElement.Base->isVirtual())
        return CharUnits::fromQuantity(-1ULL);

      if (NumPublicPaths > 1) // Won't use offsets, skip computation.
        continue;

      // Accumulate the base class offsets.
      const ASTRecordLayout &L = Ctx.getASTRecordLayout(PathElement.Class);
      Offset += L.getBaseClassOffset(
          PathElement.Base->getType()->getAsCXXRecordDecl());
    }
  }

  // -2: Src is not a public base of Dst.
  if (NumPublicPaths == 0)
    return CharUnits::fromQuantity(-2ULL);

  // -3: Src is a multiple public base type but never a virtual base type.
  if (NumPublicPaths > 1)
    return CharUnits::fromQuantity(-3ULL);

  // Otherwise, the Src type is a unique public nonvirtual base type of Dst.
  // Return the offset of Src from the origin of Dst.
  return Offset;
}

bool canUseSingleInheritance(const CXXRecordDecl *RD) {
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
  return BaseDecl->isEmpty() ||
         BaseDecl->isDynamicClass() == RD->isDynamicClass();
}

namespace {
/// Contains virtual and non-virtual bases seen when traversing a class
/// hierarchy.
struct SeenBases {
  llvm::SmallPtrSet<const CXXRecordDecl *, 16> NonVirtualBases;
  llvm::SmallPtrSet<const CXXRecordDecl *, 16> VirtualBases;
};
} // namespace

static unsigned computeVMIClassTypeInfoFlags(const CXXBaseSpecifier *Base,
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
    Flags |= computeVMIClassTypeInfoFlags(&I, Bases);

  return Flags;
}

unsigned computeVMIClassTypeInfoFlags(const CXXRecordDecl *RD) {
  unsigned Flags = 0;
  SeenBases Bases;

  // Walk all bases.
  for (const auto &I : RD->bases())
    Flags |= computeVMIClassTypeInfoFlags(&I, Bases);

  return Flags;
}

/// Returns whether the given record type is incomplete.
static bool isIncompleteClassType(const RecordType *RecordTy) {
  return !RecordTy->getDecl()->getDefinitionOrSelf()->isCompleteDefinition();
}

bool containsIncompleteClassType(QualType Ty) {
  if (const auto *RecordTy = dyn_cast<RecordType>(Ty)) {
    if (isIncompleteClassType(RecordTy))
      return true;
  }

  if (const auto *PointerTy = dyn_cast<PointerType>(Ty))
    return containsIncompleteClassType(PointerTy->getPointeeType());

  if (const auto *MemberPointerTy = dyn_cast<MemberPointerType>(Ty)) {
    // Check if the class type is incomplete.
    if (!MemberPointerTy->getMostRecentCXXRecordDecl()->hasDefinition())
      return true;

    return containsIncompleteClassType(MemberPointerTy->getPointeeType());
  }

  return false;
}

unsigned extractPBaseFlags(const ASTContext &Ctx, QualType &Type) {
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
  if (containsIncompleteClassType(Type))
    Flags |= PTI_Incomplete;

  if (auto *Proto = Type->getAs<FunctionProtoType>()) {
    if (Proto->isNothrow()) {
      Flags |= PTI_Noexcept;
      Type = Ctx.getFunctionTypeWithExceptionSpec(Type, EST_None);
    }
  }

  return Flags;
}

} // namespace clang::CodeGenUtils
