//===--- StructPackAlignCheck.cpp - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StructPackAlignCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <cmath>

using namespace clang::ast_matchers;

namespace clang::tidy::altera {

void StructPackAlignCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(recordDecl(isStruct(), isDefinition(),
                                unless(isExpansionInSystemHeader()))
                         .bind("struct"),
                     this);
}

CharUnits
StructPackAlignCheck::computeRecommendedAlignment(CharUnits MinByteSize) const {
  CharUnits NewAlign = CharUnits::fromQuantity(1);
  if (!MinByteSize.isPowerOfTwo()) {
    CharUnits::QuantityType MSB = MinByteSize.getQuantity();
    for (; MSB > 0; MSB /= 2) {
      NewAlign =
          NewAlign.alignTo(CharUnits::fromQuantity(NewAlign.getQuantity() * 2));
      // Abort if the computed alignment meets the maximum configured alignment.
      if (NewAlign.getQuantity() >= MaxConfiguredAlignment)
        break;
    }
  } else {
    NewAlign = MinByteSize;
  }
  return NewAlign;
}

void StructPackAlignCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Struct = Result.Nodes.getNodeAs<RecordDecl>("struct");

  // Do not trigger on templated struct declarations because the packing and
  // alignment requirements are unknown.
  if (Struct->isTemplated())
     return;

  // Packing and alignment requirements for invalid decls are meaningless.
  if (Struct->isInvalidDecl())
    return;

  // Get sizing info for the struct.
  llvm::SmallVector<std::pair<unsigned int, unsigned int>, 10> FieldSizes;
  unsigned int TotalBitSize = 0;
  for (const FieldDecl *StructField : Struct->fields()) {
    // For each StructField, record how big it is (in bits).
    // Would be good to use a pair of <offset, size> to advise a better
    // packing order.
    QualType StructFieldTy = StructField->getType();
    if (StructFieldTy->isIncompleteType())
      return;
    unsigned int StructFieldWidth =
        (unsigned int)Result.Context->getTypeInfo(StructFieldTy.getTypePtr())
            .Width;
    FieldSizes.emplace_back(StructFieldWidth, StructField->getFieldIndex());
    // FIXME: Recommend a reorganization of the struct (sort by StructField
    // size, largest to smallest).
    TotalBitSize += StructFieldWidth;
  }

  uint64_t CharSize = Result.Context->getCharWidth();
  CharUnits CurrSize = Result.Context->getASTRecordLayout(Struct).getSize();
  CharUnits MinByteSize =
      CharUnits::fromQuantity(std::max<clang::CharUnits::QuantityType>(
          ceil(static_cast<float>(TotalBitSize) / CharSize), 1));
  CharUnits MaxAlign = CharUnits::fromQuantity(
      ceil((float)Struct->getMaxAlignment() / CharSize));
  CharUnits CurrAlign =
      Result.Context->getASTRecordLayout(Struct).getAlignment();
  CharUnits NewAlign = computeRecommendedAlignment(MinByteSize);

  bool IsPacked = Struct->hasAttr<PackedAttr>();
  bool NeedsPacking = (MinByteSize < CurrSize) && (MaxAlign != NewAlign) &&
                      (CurrSize != NewAlign);
  bool NeedsAlignment = CurrAlign.getQuantity() != NewAlign.getQuantity();

  if (!NeedsAlignment && !NeedsPacking)
    return;

  // If it's using much more space than it needs, suggest packing.
  // (Do not suggest packing if it is currently explicitly aligned to what the
  // minimum byte size would suggest as the new alignment.)
  if (NeedsPacking && !IsPacked) {
    diag(Struct->getLocation(),
         "accessing fields in struct %0 is inefficient due to padding; only "
         "needs %1 bytes but is using %2 bytes")
        << Struct << (int)MinByteSize.getQuantity()
        << (int)CurrSize.getQuantity()
        << FixItHint::CreateInsertion(Struct->getEndLoc().getLocWithOffset(1),
                                      " __attribute__((packed))");
    diag(Struct->getLocation(),
         "use \"__attribute__((packed))\" to reduce the amount of padding "
         "applied to struct %0",
         DiagnosticIDs::Note)
        << Struct;
  }

  FixItHint FixIt;
  AlignedAttr *Attribute = Struct->getAttr<AlignedAttr>();
  std::string NewAlignQuantity = std::to_string((int)NewAlign.getQuantity());
  if (Attribute) {
    FixIt = FixItHint::CreateReplacement(
        Attribute->getRange(),
        (Twine("aligned(") + NewAlignQuantity + ")").str());
  } else {
    FixIt = FixItHint::CreateInsertion(
        Struct->getEndLoc().getLocWithOffset(1),
        (Twine(" __attribute__((aligned(") + NewAlignQuantity + ")))").str());
  }

  // And suggest the minimum power-of-two alignment for the struct as a whole
  // (with and without packing).
  if (NeedsAlignment) {
    diag(Struct->getLocation(),
         "accessing fields in struct %0 is inefficient due to poor alignment; "
         "currently aligned to %1 bytes, but recommended alignment is %2 bytes")
        << Struct << (int)CurrAlign.getQuantity() << NewAlignQuantity << FixIt;

    diag(Struct->getLocation(),
         "use \"__attribute__((aligned(%0)))\" to align struct %1 to %0 bytes",
         DiagnosticIDs::Note)
        << NewAlignQuantity << Struct;
  }
}

void StructPackAlignCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "MaxConfiguredAlignment", MaxConfiguredAlignment);
}

} // namespace clang::tidy::altera
