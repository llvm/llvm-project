//===- DialectSMT.cpp - Pybind module for SMT dialect API support ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Bindings/Python/NanobindUtils.h"

#include "aiir-c/Dialect/SMT.h"
#include "aiir-c/IR.h"
#include "aiir-c/Support.h"
#include "aiir-c/Target/ExportSMTLIB.h"
#include "aiir/Bindings/Python/Diagnostics.h"
#include "aiir/Bindings/Python/IRCore.h"
#include "aiir/Bindings/Python/Nanobind.h"
#include "aiir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

using namespace nanobind::literals;
using namespace aiir;
using namespace aiir::python::nanobind_adaptors;

namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {
namespace smt {
struct BoolType : PyConcreteType<BoolType> {
  static constexpr IsAFunctionTy isaFunction = aiirSMTTypeIsABool;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirSMTBoolTypeGetTypeID;
  static constexpr const char *pyClassName = "BoolType";
  static inline const AiirStringRef name = aiirSMTBoolTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyAiirContext context) {
          return BoolType(context->getRef(),
                          aiirSMTTypeGetBool(context.get()->get()));
        },
        nb::arg("context").none() = nb::none());
  }
};

struct BitVectorType : PyConcreteType<BitVectorType> {
  static constexpr IsAFunctionTy isaFunction = aiirSMTTypeIsABitVector;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirSMTBitVectorTypeGetTypeID;
  static constexpr const char *pyClassName = "BitVectorType";
  static inline const AiirStringRef name = aiirSMTBitVectorTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t width, DefaultingPyAiirContext context) {
          return BitVectorType(
              context->getRef(),
              aiirSMTTypeGetBitVector(context.get()->get(), width));
        },
        nb::arg("width"), nb::arg("context").none() = nb::none());
  }
};

struct IntType : PyConcreteType<IntType> {
  static constexpr IsAFunctionTy isaFunction = aiirSMTTypeIsAInt;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirSMTIntTypeGetTypeID;
  static constexpr const char *pyClassName = "IntType";
  static inline const AiirStringRef name = aiirSMTIntTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyAiirContext context) {
          return IntType(context->getRef(),
                         aiirSMTTypeGetInt(context.get()->get()));
        },
        nb::arg("context").none() = nb::none());
  }
};

static void populateDialectSMTSubmodule(nanobind::module_ &m) {
  BoolType::bind(m);
  BitVectorType::bind(m);
  IntType::bind(m);

  auto exportSMTLIB = [](AiirOperation module, bool inlineSingleUseValues,
                         bool indentLetBody, bool emitReset) {
    CollectDiagnosticsToStringScope scope(aiirOperationGetContext(module));
    PyPrintAccumulator printAccum;
    AiirLogicalResult result = aiirTranslateOperationToSMTLIB(
        module, printAccum.getCallback(), printAccum.getUserData(),
        inlineSingleUseValues, indentLetBody, emitReset);
    if (aiirLogicalResultIsSuccess(result))
      return printAccum.join();
    throw nb::value_error(
        ("Failed to export smtlib.\nDiagnostic message " + scope.takeMessage())
            .c_str());
  };

  m.def(
      "export_smtlib",
      [&exportSMTLIB](const PyOperation &module, bool inlineSingleUseValues,
                      bool indentLetBody, bool emitReset) {
        return exportSMTLIB(module, inlineSingleUseValues, indentLetBody,
                            emitReset);
      },
      "module"_a, "inline_single_use_values"_a = false,
      "indent_let_body"_a = false, "emit_reset"_a = true);
  m.def(
      "export_smtlib",
      [&exportSMTLIB](PyModule &module, bool inlineSingleUseValues,
                      bool indentLetBody, bool emitReset) {
        return exportSMTLIB(aiirModuleGetOperation(module.get()),
                            inlineSingleUseValues, indentLetBody, emitReset);
      },
      "module"_a, "inline_single_use_values"_a = false,
      "indent_let_body"_a = false, "emit_reset"_a = true);
}
} // namespace smt
} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir

NB_MODULE(_aiirDialectsSMT, m) {
  m.doc() = "AIIR SMT Dialect";

  python::AIIR_BINDINGS_PYTHON_DOMAIN::smt::populateDialectSMTSubmodule(m);
}
