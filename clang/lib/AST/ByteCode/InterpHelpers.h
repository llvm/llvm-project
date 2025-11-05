//===--- InterpHelpers.h - Interpreter Helper Functions --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_INTERPHELPERS_H
#define LLVM_CLANG_AST_INTERP_INTERPHELPERS_H

#include "DynamicAllocator.h"
#include "InterpState.h"
#include "Pointer.h"

namespace clang {
class CallExpr;
class OffsetOfExpr;

namespace interp {
class Block;
struct Descriptor;

/// Interpreter entry point.
bool Interpret(InterpState &S);

/// Interpret a builtin function.
bool InterpretBuiltin(InterpState &S, CodePtr OpPC, const CallExpr *Call,
                      uint32_t BuiltinID);

/// Interpret an offsetof operation.
bool InterpretOffsetOf(InterpState &S, CodePtr OpPC, const OffsetOfExpr *E,
                       ArrayRef<int64_t> ArrayIndices, int64_t &Result);

/// Checks if the array is offsetable.
bool CheckArray(InterpState &S, CodePtr OpPC, const Pointer &Ptr);

/// Checks if a pointer is live and accessible.
bool CheckLive(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
               AccessKinds AK);

/// Checks if a pointer is a dummy pointer.
bool CheckDummy(InterpState &S, CodePtr OpPC, const Block *B, AccessKinds AK);

/// Checks if a pointer is in range.
bool CheckRange(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                AccessKinds AK);

/// Checks if a field from which a pointer is going to be derived is valid.
bool CheckRange(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                CheckSubobjectKind CSK);

/// Checks if a pointer points to a mutable field.
bool CheckMutable(InterpState &S, CodePtr OpPC, const Pointer &Ptr);

/// Checks if a value can be loaded from a block.
bool CheckLoad(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
               AccessKinds AK = AK_Read);

/// Diagnose mismatched new[]/delete or new/delete[] pairs.
bool CheckNewDeleteForms(InterpState &S, CodePtr OpPC,
                         DynamicAllocator::Form AllocForm,
                         DynamicAllocator::Form DeleteForm, const Descriptor *D,
                         const Expr *NewExpr);

/// Copy the contents of Src into Dest.
bool DoMemcpy(InterpState &S, CodePtr OpPC, const Pointer &Src, Pointer &Dest);

template <typename T>
static bool handleOverflow(InterpState &S, CodePtr OpPC, const T &SrcValue) {
  const Expr *E = S.Current->getExpr(OpPC);
  S.CCEDiag(E, diag::note_constexpr_overflow) << SrcValue << E->getType();
  return S.noteUndefinedBehavior();
}

inline bool CheckArraySize(InterpState &S, CodePtr OpPC, uint64_t NumElems) {
  uint64_t Limit = S.getLangOpts().ConstexprStepLimit;
  if (Limit != 0 && NumElems > Limit) {
    S.FFDiag(S.Current->getSource(OpPC),
             diag::note_constexpr_new_exceeds_limits)
        << NumElems << Limit;
    return false;
  }
  return true;
}

static inline llvm::RoundingMode getRoundingMode(FPOptions FPO) {
  auto RM = FPO.getRoundingMode();
  if (RM == llvm::RoundingMode::Dynamic)
    return llvm::RoundingMode::NearestTiesToEven;
  return RM;
}

inline bool Invalid(InterpState &S, CodePtr OpPC) {
  const SourceLocation &Loc = S.Current->getLocation(OpPC);
  S.FFDiag(Loc, diag::note_invalid_subexpr_in_const_expr)
      << S.Current->getRange(OpPC);
  return false;
}

template <typename SizeT>
bool CheckArraySize(InterpState &S, CodePtr OpPC, SizeT *NumElements,
                    unsigned ElemSize, bool IsNoThrow) {
  // FIXME: Both the SizeT::from() as well as the
  // NumElements.toAPSInt() in this function are rather expensive.

  // Can't be too many elements if the bitwidth of NumElements is lower than
  // that of Descriptor::MaxArrayElemBytes.
  if ((NumElements->bitWidth() - NumElements->isSigned()) <
      (sizeof(Descriptor::MaxArrayElemBytes) * 8))
    return true;

  // FIXME: GH63562
  // APValue stores array extents as unsigned,
  // so anything that is greater that unsigned would overflow when
  // constructing the array, we catch this here.
  SizeT MaxElements = SizeT::from(Descriptor::MaxArrayElemBytes / ElemSize);
  assert(MaxElements.isPositive());
  if (NumElements->toAPSInt().getActiveBits() >
          ConstantArrayType::getMaxSizeBits(S.getASTContext()) ||
      *NumElements > MaxElements) {
    if (!IsNoThrow) {
      const SourceInfo &Loc = S.Current->getSource(OpPC);

      if (NumElements->isSigned() && NumElements->isNegative()) {
        S.FFDiag(Loc, diag::note_constexpr_new_negative)
            << NumElements->toDiagnosticString(S.getASTContext());
      } else {
        S.FFDiag(Loc, diag::note_constexpr_new_too_large)
            << NumElements->toDiagnosticString(S.getASTContext());
      }
    }
    return false;
  }
  return true;
}

} // namespace interp
} // namespace clang

#endif // LLVM_CLANG_AST_INTERP_INTERPHELPERS_H
