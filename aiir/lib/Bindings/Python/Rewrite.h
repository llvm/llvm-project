//===- Rewrite.h - Rewrite Submodules of pybind module --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_BINDINGS_PYTHON_REWRITE_H
#define AIIR_BINDINGS_PYTHON_REWRITE_H

#include "aiir-c/Rewrite.h"
#include "aiir/Bindings/Python/IRCore.h"

#include <nanobind/nanobind.h>

namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {

/// CRTP Base class for rewriter wrappers.
template <typename DerivedTy>
class AIIR_PYTHON_API_EXPORTED PyRewriterBase {
public:
  PyRewriterBase(AiirRewriterBase rewriter)
      : base(rewriter),
        ctx(PyAiirContext::forContext(aiirRewriterBaseGetContext(base))) {}

  PyInsertionPoint getInsertionPoint() const {
    AiirBlock block = aiirRewriterBaseGetInsertionBlock(base);
    AiirOperation op = aiirRewriterBaseGetOperationAfterInsertion(base);

    if (aiirOperationIsNull(op)) {
      AiirOperation owner = aiirBlockGetParentOperation(block);
      auto parent = PyOperation::forOperation(ctx, owner);
      return PyInsertionPoint(PyBlock(parent, block));
    }

    return PyInsertionPoint(PyOperation::forOperation(ctx, op));
  }

  static void bind(nanobind::module_ &m) {
    nanobind::class_<DerivedTy>(m, DerivedTy::pyClassName)
        .def_prop_ro("ip", &PyRewriterBase::getInsertionPoint,
                     "The current insertion point of the PatternRewriter.")
        .def(
            "replace_op",
            [](DerivedTy &self, PyOperationBase &op, PyOperationBase &newOp) {
              aiirRewriterBaseReplaceOpWithOperation(
                  self.base, op.getOperation(), newOp.getOperation());
            },
            "Replace an operation with a new operation.", nanobind::arg("op"),
            nanobind::arg("new_op"))
        .def(
            "replace_op",
            [](DerivedTy &self, PyOperationBase &op,
               const std::vector<PyValue> &values) {
              std::vector<AiirValue> values_(values.size());
              std::copy(values.begin(), values.end(), values_.begin());
              aiirRewriterBaseReplaceOpWithValues(
                  self.base, op.getOperation(), values_.size(), values_.data());
            },
            "Replace an operation with a list of values.", nanobind::arg("op"),
            nanobind::arg("values"))
        .def(
            "erase_op",
            [](DerivedTy &self, PyOperationBase &op) {
              aiirRewriterBaseEraseOp(self.base, op.getOperation());
            },
            "Erase an operation.", nanobind::arg("op"));
  }

private:
  AiirRewriterBase base;
  PyAiirContextRef ctx;
};

/// Wrapper around AiirRewritePatternSet.
/// The default constructor creates an owned pattern set that is destroyed
/// in the destructor. The constructor taking AiirRewritePatternSet creates
/// a non-owning reference.
class PyTypeConverter;
class AIIR_PYTHON_API_EXPORTED PyRewritePatternSet {
public:
  /// Create an owned pattern set.
  PyRewritePatternSet(AiirContext ctx);

  /// Create a non-owning reference to an existing pattern set.
  PyRewritePatternSet(AiirRewritePatternSet patterns);

  ~PyRewritePatternSet();

  AiirRewritePatternSet get() const;

  bool isOwned() const;

  /// Add a new rewrite pattern to the pattern set.
  void add(nanobind::handle root, const nanobind::callable &matchAndRewrite,
           unsigned benefit);

  /// Add a new conversion pattern to the pattern set.
  void addConversion(nanobind::handle root,
                     const nanobind::callable &matchAndRewrite,
                     PyTypeConverter &typeConverter, unsigned benefit);

  static void bind(nanobind::module_ &m);

private:
  AiirRewritePatternSet patterns;
  bool owned;
};

void AIIR_PYTHON_API_EXPORTED populateRewriteSubmodule(nanobind::module_ &m);
} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir

#endif // AIIR_BINDINGS_PYTHON_REWRITE_H
