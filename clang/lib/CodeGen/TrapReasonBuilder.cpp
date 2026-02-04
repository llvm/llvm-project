//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements TrapReasonBuilder and related classes.
///
//===----------------------------------------------------------------------===//
#include "TrapReasonBuilder.h"

namespace clang {
namespace CodeGen {

TrapReasonBuilder::TrapReasonBuilder(DiagnosticsEngine *DiagObj,
                                     unsigned DiagID, TrapReason &TR)
    : DiagnosticBuilder(DiagObj, SourceLocation(), DiagID), TR(TR) {
  assert(DiagObj->getDiagnosticIDs()->isTrapDiag(DiagID));
}

TrapReasonBuilder::~TrapReasonBuilder() {
  // Store the trap message and category into the TrapReason object.
  getMessage(TR.Message);
  TR.Category = getCategory();

  // Make sure that when `DiagnosticBuilder::~DiagnosticBuilder()`
  // calls `Emit()` that it does nothing.
  Clear();
}

void TrapReasonBuilder::getMessage(SmallVectorImpl<char> &Storage) {
  // Render the Diagnostic
  Diagnostic Info(getDiagnosticsEngine(), *this);
  Info.FormatDiagnostic(Storage);
}

StringRef TrapReasonBuilder::getCategory() {
  auto CategoryID =
      getDiagnosticsEngine()->getDiagnosticIDs()->getCategoryNumberForDiag(
          getDiagID());
  if (CategoryID == 0)
    return "";
  return getDiagnosticsEngine()->getDiagnosticIDs()->getCategoryNameFromID(
      CategoryID);
}
} // namespace CodeGen
} // namespace clang
