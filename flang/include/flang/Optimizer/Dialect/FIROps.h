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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace fir {

class FirEndOp;
class DoLoopOp;
class RealAttr;

void buildCmpCOp(mlir::OpBuilder &builder, mlir::OperationState &result,
                 mlir::arith::CmpFPredicate predicate, mlir::Value lhs,
                 mlir::Value rhs);
unsigned getCaseArgumentOffset(llvm::ArrayRef<mlir::Attribute> cases,
                               unsigned dest);
DoLoopOp getForInductionVarOwner(mlir::Value val);
mlir::ParseResult isValidCaseAttr(mlir::Attribute attr);
mlir::ParseResult parseCmpcOp(mlir::OpAsmParser &parser,
                              mlir::OperationState &result);
mlir::ParseResult parseSelector(mlir::OpAsmParser &parser,
                                mlir::OperationState &result,
                                mlir::OpAsmParser::UnresolvedOperand &selector,
                                mlir::Type &type);
bool useStrictVolatileVerification();

static constexpr llvm::StringRef getNormalizedLowerBoundAttrName() {
  return "normalized.lb";
}

/// Model operations which affect global debugging information
struct DebuggingResource
    : public mlir::SideEffects::Resource::Base<DebuggingResource> {
  mlir::StringRef getName() final { return "DebuggingResource"; }
};

/// Model operations which read from/write to volatile memory
struct VolatileMemoryResource
    : public mlir::SideEffects::Resource::Base<VolatileMemoryResource> {
  mlir::StringRef getName() final { return "VolatileMemoryResource"; }
};

class CoordinateIndicesAdaptor;
using IntOrValue = llvm::PointerUnion<mlir::IntegerAttr, mlir::Value>;

} // namespace fir

#define GET_OP_CLASSES
#include "flang/Optimizer/Dialect/FIROps.h.inc"

namespace fir {
class CoordinateIndicesAdaptor {
public:
  using value_type = IntOrValue;

  CoordinateIndicesAdaptor(mlir::DenseI32ArrayAttr fieldIndices,
                           mlir::ValueRange values)
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
             llvm::detail::IterOfRange<const mlir::ValueRange> valuesIter)
        : base(base), fieldIter(fieldIter), valuesIter(valuesIter) {}

    value_type operator*() const {
      if (fieldIter && **fieldIter != fir::CoordinateOp::kDynamicIndex) {
        return mlir::IntegerAttr::get(base->fieldIndices.getElementType(),
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
    llvm::detail::IterOfRange<const mlir::ValueRange> valuesIter;
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
  mlir::DenseI32ArrayAttr fieldIndices;
  mlir::ValueRange values;
};

struct LocalitySpecifierOperands {
  llvm::SmallVector<::mlir::Value> privateVars;
  llvm::SmallVector<::mlir::Attribute> privateSyms;
};
} // namespace fir

#endif // FORTRAN_OPTIMIZER_DIALECT_FIROPS_H
