//===-- Optimizer/Dialect/FIROps.h - FIR operations -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_DIALECT_FIROPS_H
#define FORTRAN_OPTIMIZER_DIALECT_FIROPS_H

#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/FirAliasTagOpInterface.h"
#include "flang/Optimizer/Dialect/FortranVariableInterface.h"
#include "flang/Optimizer/Dialect/SafeTempArrayCopyAttrInterface.h"
#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/LLVMIR/LLVMAttrs.h"
#include "aiir/Interfaces/LoopLikeInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"
#include "aiir/Interfaces/ViewLikeInterface.h"

namespace fir {

class FirEndOp;
class DoLoopOp;
class RealAttr;

void buildCmpCOp(aiir::OpBuilder &builder, aiir::OperationState &result,
                 aiir::arith::CmpFPredicate predicate, aiir::Value lhs,
                 aiir::Value rhs);
unsigned getCaseArgumentOffset(llvm::ArrayRef<aiir::Attribute> cases,
                               unsigned dest);
DoLoopOp getForInductionVarOwner(aiir::Value val);
aiir::ParseResult isValidCaseAttr(aiir::Attribute attr);
aiir::ParseResult parseCmpcOp(aiir::OpAsmParser &parser,
                              aiir::OperationState &result);
aiir::ParseResult parseSelector(aiir::OpAsmParser &parser,
                                aiir::OperationState &result,
                                aiir::OpAsmParser::UnresolvedOperand &selector,
                                aiir::Type &type);
bool useStrictVolatileVerification();

static constexpr llvm::StringRef getNormalizedLowerBoundAttrName() {
  return "normalized.lb";
}

/// Model operations which affect global debugging information
struct DebuggingResource
    : public aiir::SideEffects::Resource::Base<DebuggingResource> {
  aiir::StringRef getName() const final { return "DebuggingResource"; }
  bool isAddressable() const override { return false; }
};

/// Model operations which read from/write to volatile memory
struct VolatileMemoryResource
    : public aiir::SideEffects::Resource::Base<VolatileMemoryResource> {
  aiir::StringRef getName() const final { return "VolatileMemoryResource"; }
  bool isAddressable() const override { return false; }
};

class CoordinateIndicesAdaptor;
using IntOrValue = llvm::PointerUnion<aiir::IntegerAttr, aiir::Value>;

} // namespace fir

#define GET_OP_CLASSES
#include "flang/Optimizer/Dialect/FIROps.h.inc"

namespace fir {
class CoordinateIndicesAdaptor {
public:
  using value_type = IntOrValue;

  CoordinateIndicesAdaptor(aiir::DenseI32ArrayAttr fieldIndices,
                           aiir::ValueRange values)
      : fieldIndices(fieldIndices), values(values) {}

  value_type operator[](size_t index) const {
    assert(index < size() && "index out of bounds");
    return *std::next(begin(), index);
  }

  size_t size() const {
    return fieldIndices ? fieldIndices.size() : values.size();
  }

  bool empty() const {
    return values.empty() && (!fieldIndices || fieldIndices.empty());
  }

  class iterator
      : public llvm::iterator_facade_base<iterator, std::forward_iterator_tag,
                                          value_type, std::ptrdiff_t,
                                          value_type *, value_type> {
  public:
    iterator(const CoordinateIndicesAdaptor *base,
             std::optional<llvm::ArrayRef<int32_t>::iterator> fieldIter,
             llvm::detail::IterOfRange<const aiir::ValueRange> valuesIter)
        : base(base), fieldIter(fieldIter), valuesIter(valuesIter) {}

    value_type operator*() const {
      if (fieldIter && **fieldIter != fir::CoordinateOp::kDynamicIndex) {
        return aiir::IntegerAttr::get(base->fieldIndices.getElementType(),
                                      **fieldIter);
      }
      return *valuesIter;
    }

    iterator &operator++() {
      if (fieldIter) {
        if (**fieldIter == fir::CoordinateOp::kDynamicIndex)
          valuesIter++;
        (*fieldIter)++;
      } else {
        valuesIter++;
      }
      return *this;
    }

    bool operator==(const iterator &rhs) const {
      return base == rhs.base && fieldIter == rhs.fieldIter &&
             valuesIter == rhs.valuesIter;
    }

  private:
    const CoordinateIndicesAdaptor *base;
    std::optional<llvm::ArrayRef<int32_t>::const_iterator> fieldIter;
    llvm::detail::IterOfRange<const aiir::ValueRange> valuesIter;
  };

  iterator begin() const {
    std::optional<llvm::ArrayRef<int32_t>::const_iterator> fieldIter;
    if (fieldIndices)
      fieldIter = fieldIndices.asArrayRef().begin();
    return iterator(this, fieldIter, values.begin());
  }

  iterator end() const {
    std::optional<llvm::ArrayRef<int32_t>::const_iterator> fieldIter;
    if (fieldIndices)
      fieldIter = fieldIndices.asArrayRef().end();
    return iterator(this, fieldIter, values.end());
  }

private:
  aiir::DenseI32ArrayAttr fieldIndices;
  aiir::ValueRange values;
};

struct LocalitySpecifierOperands {
  llvm::SmallVector<::aiir::Value> privateVars;
  llvm::SmallVector<::aiir::Attribute> privateSyms;
};

/// Returns true if the given box value may be absent.
/// The given value must have BaseBoxType.
bool mayBeAbsentBox(aiir::Value val);

} // namespace fir

#endif // FORTRAN_OPTIMIZER_DIALECT_FIROPS_H
