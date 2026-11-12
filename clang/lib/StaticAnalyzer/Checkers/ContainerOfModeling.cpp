//===- ContainerOfModeling.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/ContainerOfModeling.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicExtent.h"
#include "llvm/Support/CheckedArithmetic.h"

namespace clang::ento {

static QualType getRegionObjectType(const MemRegion *Region) {
  if (const auto *TVR = dyn_cast<TypedValueRegion>(Region))
    return TVR->getValueType();
  if (const auto *SR = dyn_cast<SymbolicRegion>(Region))
    return SR->getPointeeStaticType();
  return {};
}

/// Return true when the region containing \p ContainerRegion has type
/// \p ContainerType. ElementRegion represents both array elements and casts,
/// so the type of ContainerRegion itself is not sufficient evidence.
static bool hasContainerTypeProvenance(const SubRegion *ContainerRegion,
                                       QualType ContainerType,
                                       ASTContext &Ctx) {
  const MemRegion *StorageRegion = ContainerRegion;
  if (const auto *ER = dyn_cast<ElementRegion>(ContainerRegion)) {
    if (!ASTContext::hasSameUnqualifiedType(ER->getElementType(),
                                            ContainerType))
      return false;
    StorageRegion = ER->getSuperRegion();
  }

  QualType StorageType = getRegionObjectType(StorageRegion);
  if (StorageType.isNull())
    return false;

  if (const ArrayType *AT = Ctx.getAsArrayType(StorageType))
    StorageType = AT->getElementType();

  return ASTContext::hasSameUnqualifiedType(StorageType, ContainerType);
}

/// Return whether the concrete storage containing \p ContainerRegion is large
/// enough to contain an object of \p ContainerType at that region's offset.
/// Return std::nullopt when either the offset or the extent is symbolic.
static std::optional<bool>
hasSufficientContainerExtent(ProgramStateRef State,
                             const SubRegion *ContainerRegion,
                             QualType ContainerType, SValBuilder &SVB) {
  ASTContext &Ctx = SVB.getContext();
  RegionOffset Offset = ContainerRegion->getAsOffset();
  if (!Offset.isValid() || Offset.hasSymbolicOffset())
    return std::nullopt;

  const int64_t OffsetBits = Offset.getOffset();
  const uint64_t CharWidth = Ctx.getCharWidth();
  if (OffsetBits < 0 || static_cast<uint64_t>(OffsetBits) % CharWidth != 0)
    return false;

  const MemRegion *BaseRegion = Offset.getRegion();
  const auto BaseExtent =
      getDynamicExtent(State, BaseRegion, SVB).getAs<nonloc::ConcreteInt>();
  if (!BaseExtent)
    return std::nullopt;

  const int64_t ContainerSize =
      Ctx.getTypeSizeInChars(ContainerType).getQuantity();
  if (ContainerSize < 0)
    return false;

  const uint64_t OffsetChars = static_cast<uint64_t>(OffsetBits) / CharWidth;
  const uint64_t ContainerSizeChars = static_cast<uint64_t>(ContainerSize);
  if (OffsetChars > std::numeric_limits<uint64_t>::max() - ContainerSizeChars)
    return false;

  const uint64_t RequiredExtent = OffsetChars + ContainerSizeChars;
  const llvm::APSInt RequiredExtentValue =
      llvm::APSInt::getUnsigned(RequiredExtent);
  return llvm::APSInt::compareValues(*BaseExtent->getValue(),
                                     RequiredExtentValue) >= 0;
}

const SubRegion *getContainerOfParentRegion(const ElementRegion *ContainerER,
                                            ProgramStateRef State,
                                            SValBuilder &SVB) {
  ASTContext &Ctx = SVB.getContext();
  const MemRegion *SuperRegion = ContainerER->getSuperRegion();
  const FieldRegion *FieldR = nullptr;
  int64_t CharacterIndex = 0;

  if (const auto *CharacterER = dyn_cast<ElementRegion>(SuperRegion)) {
    QualType CharacterType = CharacterER->getElementType();
    if (!CharacterType->isCharType())
      return nullptr;

    const auto ConcreteIndex =
        CharacterER->getIndex().getAs<nonloc::ConcreteInt>();
    if (!ConcreteIndex)
      return nullptr;

    std::optional<int64_t> Index = ConcreteIndex->getValue()->tryExtValue();
    if (!Index)
      return nullptr;
    CharacterIndex = *Index;

    FieldR = dyn_cast<FieldRegion>(CharacterER->getSuperRegion());
  } else {
    // SValBuilder folds an adjustment of zero, so a first field is represented
    // without an intermediate character ElementRegion.
    FieldR = dyn_cast<FieldRegion>(SuperRegion);
  }

  if (!FieldR)
    return nullptr;

  const FieldDecl *Field = FieldR->getDecl();
  if (Field->isBitField())
    return nullptr;

  QualType ContainerType =
      ContainerER->getElementType().getCanonicalType().getUnqualifiedType();
  const auto *ContainerRT = ContainerType->getAs<RecordType>();
  if (!ContainerRT)
    return nullptr;

  const RecordDecl *FieldParent = Field->getParent();
  if (!FieldParent || !FieldParent->isCompleteDefinition() ||
      ContainerRT->getDecl()->getCanonicalDecl() !=
          FieldParent->getCanonicalDecl())
    return nullptr;

  const uint64_t FieldOffsetBits = Ctx.getFieldOffset(Field);
  const uint64_t CharWidth = Ctx.getCharWidth();
  if (FieldOffsetBits % CharWidth != 0 || CharacterIndex > 0)
    return nullptr;

  // Avoid negating INT64_MIN while comparing the signed character index with
  // the unsigned ABI field offset.
  const uint64_t BackwardOffset =
      static_cast<uint64_t>(-(CharacterIndex + 1)) + 1;

  const std::optional<uint64_t> BackwardOffsetBits =
    llvm::checkedMulUnsigned(BackwardOffset, CharWidth);

  if (!BackwardOffsetBits || *BackwardOffsetBits != FieldOffsetBits)
    return nullptr;

  const auto *ParentRegion = dyn_cast<SubRegion>(FieldR->getSuperRegion());
  if (!ParentRegion)
    return nullptr;

  std::optional<bool> HasSufficientExtent =
      hasSufficientContainerExtent(State, ParentRegion, ContainerType, SVB);
  if (HasSufficientExtent == false)
    return nullptr;
  if (HasSufficientExtent == std::nullopt &&
      !hasContainerTypeProvenance(ParentRegion, ContainerType, Ctx))
    return nullptr;

  return ParentRegion;
}

} // namespace clang::ento
