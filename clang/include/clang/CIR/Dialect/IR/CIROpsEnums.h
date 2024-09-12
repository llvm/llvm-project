//===- CIROpsEnumsDialect.h - MLIR Dialect for CIR ----------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Target dialect for CIR in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CIR_CIROPSENUMS_H_
#define MLIR_DIALECT_CIR_CIROPSENUMS_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h.inc"

namespace mlir {
namespace cir {

static bool isExternalLinkage(GlobalLinkageKind Linkage) {
  return Linkage == GlobalLinkageKind::ExternalLinkage;
}
static bool isAvailableExternallyLinkage(GlobalLinkageKind Linkage) {
  return Linkage == GlobalLinkageKind::AvailableExternallyLinkage;
}
static bool isLinkOnceAnyLinkage(GlobalLinkageKind Linkage) {
  return Linkage == GlobalLinkageKind::LinkOnceAnyLinkage;
}
static bool isLinkOnceODRLinkage(GlobalLinkageKind Linkage) {
  return Linkage == GlobalLinkageKind::LinkOnceODRLinkage;
}
static bool isLinkOnceLinkage(GlobalLinkageKind Linkage) {
  return isLinkOnceAnyLinkage(Linkage) || isLinkOnceODRLinkage(Linkage);
}
static bool isWeakAnyLinkage(GlobalLinkageKind Linkage) {
  return Linkage == GlobalLinkageKind::WeakAnyLinkage;
}
static bool isWeakODRLinkage(GlobalLinkageKind Linkage) {
  return Linkage == GlobalLinkageKind::WeakODRLinkage;
}
static bool isWeakLinkage(GlobalLinkageKind Linkage) {
  return isWeakAnyLinkage(Linkage) || isWeakODRLinkage(Linkage);
}
static bool isInternalLinkage(GlobalLinkageKind Linkage) {
  return Linkage == GlobalLinkageKind::InternalLinkage;
}
static bool isPrivateLinkage(GlobalLinkageKind Linkage) {
  return Linkage == GlobalLinkageKind::PrivateLinkage;
}
static bool isLocalLinkage(GlobalLinkageKind Linkage) {
  return isInternalLinkage(Linkage) || isPrivateLinkage(Linkage);
}
static bool isExternalWeakLinkage(GlobalLinkageKind Linkage) {
  return Linkage == GlobalLinkageKind::ExternalWeakLinkage;
}
LLVM_ATTRIBUTE_UNUSED static bool isCommonLinkage(GlobalLinkageKind Linkage) {
  return Linkage == GlobalLinkageKind::CommonLinkage;
}
LLVM_ATTRIBUTE_UNUSED static bool
isValidDeclarationLinkage(GlobalLinkageKind Linkage) {
  return isExternalWeakLinkage(Linkage) || isExternalLinkage(Linkage);
}

/// Whether the definition of this global may be replaced by something
/// non-equivalent at link time. For example, if a function has weak linkage
/// then the code defining it may be replaced by different code.
LLVM_ATTRIBUTE_UNUSED static bool
isInterposableLinkage(GlobalLinkageKind Linkage) {
  switch (Linkage) {
  case GlobalLinkageKind::WeakAnyLinkage:
  case GlobalLinkageKind::LinkOnceAnyLinkage:
  case GlobalLinkageKind::CommonLinkage:
  case GlobalLinkageKind::ExternalWeakLinkage:
    return true;

  case GlobalLinkageKind::AvailableExternallyLinkage:
  case GlobalLinkageKind::LinkOnceODRLinkage:
  case GlobalLinkageKind::WeakODRLinkage:
    // The above three cannot be overridden but can be de-refined.

  case GlobalLinkageKind::ExternalLinkage:
  case GlobalLinkageKind::InternalLinkage:
  case GlobalLinkageKind::PrivateLinkage:
    return false;
  }
  llvm_unreachable("Fully covered switch above!");
}

/// Whether the definition of this global may be discarded if it is not used
/// in its compilation unit.
LLVM_ATTRIBUTE_UNUSED static bool
isDiscardableIfUnused(GlobalLinkageKind Linkage) {
  return isLinkOnceLinkage(Linkage) || isLocalLinkage(Linkage) ||
         isAvailableExternallyLinkage(Linkage);
}

/// Whether the definition of this global may be replaced at link time.  NB:
/// Using this method outside of the code generators is almost always a
/// mistake: when working at the IR level use isInterposable instead as it
/// knows about ODR semantics.
LLVM_ATTRIBUTE_UNUSED static bool isWeakForLinker(GlobalLinkageKind Linkage) {
  return Linkage == GlobalLinkageKind::WeakAnyLinkage ||
         Linkage == GlobalLinkageKind::WeakODRLinkage ||
         Linkage == GlobalLinkageKind::LinkOnceAnyLinkage ||
         Linkage == GlobalLinkageKind::LinkOnceODRLinkage ||
         Linkage == GlobalLinkageKind::CommonLinkage ||
         Linkage == GlobalLinkageKind::ExternalWeakLinkage;
}

LLVM_ATTRIBUTE_UNUSED static bool isValidLinkage(GlobalLinkageKind L) {
  return isExternalLinkage(L) || isLocalLinkage(L) || isWeakLinkage(L) ||
         isLinkOnceLinkage(L);
}

bool operator<(mlir::cir::MemOrder, mlir::cir::MemOrder) = delete;
bool operator>(mlir::cir::MemOrder, mlir::cir::MemOrder) = delete;
bool operator<=(mlir::cir::MemOrder, mlir::cir::MemOrder) = delete;
bool operator>=(mlir::cir::MemOrder, mlir::cir::MemOrder) = delete;

// Validate an integral value which isn't known to fit within the enum's range
// is a valid AtomicOrderingCABI.
template <typename Int> inline bool isValidCIRAtomicOrderingCABI(Int I) {
  return (Int)mlir::cir::MemOrder::Relaxed <= I &&
         I <= (Int)mlir::cir::MemOrder::SequentiallyConsistent;
}

} // namespace cir
} // namespace mlir

#endif // MLIR_DIALECT_CIR_CIROPSENUMS_H_
