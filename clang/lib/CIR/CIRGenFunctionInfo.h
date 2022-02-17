//==-- CIRGenFunctionInfo.h - Representation of fn argument/return types ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines CIRGenFunctionInfo and associated types used in representing the
// CIR source types and ABI-coerced types for function arguments and
// return values.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_CIRGENFUNCTIONINFO_H
#define LLVM_CLANG_CIR_CIRGENFUNCTIONINFO_H

#include "clang/AST/CanonicalType.h"

#include "llvm/ADT/FoldingSet.h"
#include "llvm/Support/TrailingObjects.h"

#include "mlir/Dialect/CIR/IR/CIRTypes.h"

namespace cir {

/// ABIArgInfo - Helper class to encapsulate information about how a specific C
/// type should be passed to or returned from a function.
class ABIArgInfo {
public:
  enum Kind : uint8_t {
    /// Direct - Pass the argument directly using the normal converted CIR type,
    /// or by coercing to another specified type stored in 'CoerceToType'). If
    /// an offset is specified (in UIntData), then the argument passed is offset
    /// by some number of bytes in the memory representation. A dummy argument
    /// is emitted before the real argument if the specified type stored in
    /// "PaddingType" is not zero.
    Direct,

    /// Extend - Valid only for integer argument types. Same as 'direct' but
    /// also emit a zer/sign extension attribute.
    Extend,

    /// Indirect - Pass the argument indirectly via a hidden pointer with the
    /// specified alignment (0 indicates default alignment) and address space.
    Indirect,

    /// IndirectAliased - Similar to Indirect, but the pointer may be to an
    /// object that is otherwise referenced. The object is known to not be
    /// modified through any other references for the duration of the call, and
    /// the callee must not itself modify the object. Because C allows parameter
    /// variables to be modified and guarantees that they have unique addresses,
    /// the callee must defensively copy the object into a local variable if it
    /// might be modified or its address might be compared. Since those are
    /// uncommon, in principle this convention allows programs to avoid copies
    /// in more situations. However, it may introduce *extra* copies if the
    /// callee fails to prove that a copy is unnecessary and the caller
    /// naturally produces an unaliased object for the argument.
    IndirectAliased,

    /// Ignore - Ignore the argument (treat as void). Useful for void and empty
    /// structs.
    Ignore,

    /// Expand - Only valid for aggregate argument types. The structure should
    /// be expanded into consecutive arguments for its constituent fields.
    /// Currently expand is only allowed on structures whose fields are all
    /// scalar types or are themselves expandable types.
    Expand,

    /// CoerceAndExpand - Only valid for aggregate argument types. The structure
    /// should be expanded into consecutive arguments corresponding to the
    /// non-array elements of the type stored in CoerceToType.
    /// Array elements in the type are assumed to be padding and skipped.
    CoerceAndExpand,

    // TODO: translate this idea to CIR! Define it for now just to ensure that
    // we can assert it not being used
    InAlloca,
    KindFirst = Direct,
    KindLast = InAlloca
  };

private:
  mlir::Type TypeData; // canHaveCoerceToType();
  union {
    mlir::Type PaddingType;                 // canHavePaddingType()
    mlir::Type UnpaddedCoerceAndExpandType; // isCoerceAndExpand()
  };
  struct DirectAttrInfo {
    unsigned Offset;
    unsigned Align;
  };
  struct IndirectAttrInfo {
    unsigned Align;
    unsigned AddrSpace;
  };
  union {
    DirectAttrInfo DirectAttr;     // isDirect() || isExtend()
    IndirectAttrInfo IndirectAttr; // isIndirect()
    unsigned AllocaFieldIndex;     // isInAlloca()
  };
  Kind TheKind;
  bool CanBeFlattened : 1; // isDirect()

  bool canHavePaddingType() const {
    return isDirect() || isExtend() || isIndirect() || isIndirectAliased() ||
           isExpand();
  }

  void setPaddingType(mlir::Type T) {
    assert(canHavePaddingType());
    PaddingType = T;
  }

public:
  ABIArgInfo(Kind K = Direct)
      : TypeData(nullptr), PaddingType(nullptr), DirectAttr{0, 0}, TheKind(K),
        CanBeFlattened(false) {}

  static ABIArgInfo getDirect(mlir::Type T = nullptr, unsigned Offset = 0,
                              mlir::Type Padding = nullptr,
                              bool CanBeFlattened = true, unsigned Align = 0) {
    auto AI = ABIArgInfo(Direct);
    AI.setCoerceToType(T);
    AI.setPaddingType(Padding);
    AI.setDirectOffset(Offset);
    AI.setDirectAlign(Align);
    AI.setCanBeFlattened(CanBeFlattened);
    return AI;
  }

  static ABIArgInfo getIgnore() { return ABIArgInfo(Ignore); }

  Kind getKind() const { return TheKind; }
  bool isDirect() const { return TheKind == Direct; }
  bool isInAlloca() const { return TheKind == InAlloca; }
  bool isExtend() const { return TheKind == Extend; }
  bool isIndirect() const { return TheKind == Indirect; }
  bool isIndirectAliased() const { return TheKind == IndirectAliased; }
  bool isExpand() const { return TheKind == Expand; }
  bool isCoerceAndExpand() const { return TheKind == CoerceAndExpand; }

  bool canHaveCoerceToType() const {
    return isDirect() || isExtend() || isCoerceAndExpand();
  }

  void setDirectOffset(unsigned Offset) {
    assert((isDirect() || isExtend()) && "Not a direct or extend kind");
    DirectAttr.Offset = Offset;
  }

  void setDirectAlign(unsigned Align) {
    assert((isDirect() || isExtend()) && "Not a direct or extend kind");
    DirectAttr.Align = Align;
  }

  void setCanBeFlattened(bool Flatten) {
    assert(isDirect() && "Invalid kind!");
    CanBeFlattened = Flatten;
  }

  mlir::Type getPaddingType() const {
    return (canHavePaddingType() ? PaddingType : nullptr);
  }

  mlir::Type getCoerceToType() const {
    assert(canHaveCoerceToType() && "Invalid kind!");
    return TypeData;
  }

  void setCoerceToType(mlir::Type T) {
    assert(canHaveCoerceToType() && "Invalid kind!");
    TypeData = T;
  }
};

} // namespace cir

#endif
