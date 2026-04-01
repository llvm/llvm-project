//===- PythonTestModuleNanobind.cpp - PythonTest dialect extension --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This is the nanobind edition of the PythonTest dialect module.
//===----------------------------------------------------------------------===//

#include "PythonTestCAPI.h"
#include "aiir-c/BuiltinAttributes.h"
#include "aiir-c/BuiltinTypes.h"
#include "aiir-c/Diagnostics.h"
#include "aiir-c/IR.h"
#include "aiir/Bindings/Python/Diagnostics.h"
#include "aiir/Bindings/Python/IRCore.h"
#include "aiir/Bindings/Python/IRTypes.h"
#include "aiir/Bindings/Python/Nanobind.h"
#include "aiir/Bindings/Python/NanobindAdaptors.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;
using namespace aiir::python::nanobind_adaptors;
namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {
namespace python_test {
static bool aiirTypeIsARankedIntegerTensor(AiirType t) {
  return aiirTypeIsARankedTensor(t) &&
         aiirTypeIsAInteger(aiirShapedTypeGetElementType(t));
}

struct PyTestType : PyConcreteType<PyTestType> {
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAPythonTestTestType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirPythonTestTestTypeGetTypeID;
  static constexpr const char *pyClassName = "TestType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyAiirContext context) {
          return PyTestType(context->getRef(),
                            aiirPythonTestTestTypeGet(context.get()->get()));
        },
        nb::arg("context").none() = nb::none());
  }
};

struct PyTestIntegerRankedTensorType
    : PyConcreteType<PyTestIntegerRankedTensorType, PyRankedTensorType> {
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsARankedIntegerTensor;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirRankedTensorTypeGetTypeID;
  static constexpr const char *pyClassName = "TestIntegerRankedTensorType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::vector<int64_t> shape, unsigned width,
           DefaultingPyAiirContext ctx) {
          AiirAttribute encoding = aiirAttributeGetNull();
          return PyTestIntegerRankedTensorType(
              ctx->getRef(),
              aiirRankedTensorTypeGet(
                  shape.size(), shape.data(),
                  aiirIntegerTypeGet(ctx.get()->get(), width), encoding));
        },
        nb::arg("shape"), nb::arg("width"),
        nb::arg("context").none() = nb::none());
  }
};

struct PyTestTensorValue : PyConcreteValue<PyTestTensorValue> {
  static constexpr IsAFunctionTy isaFunction =
      aiirTypeIsAPythonTestTestTensorValue;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirRankedTensorTypeGetTypeID;
  static constexpr const char *pyClassName = "TestTensorValue";
  using PyConcreteValue::PyConcreteValue;

  static void bindDerived(ClassTy &c) {
    c.def("is_null", [](AiirValue &self) { return aiirValueIsNull(self); });
  }
};

class PyTestAttr : public PyConcreteAttribute<PyTestAttr> {
public:
  static constexpr IsAFunctionTy isaFunction =
      aiirAttributeIsAPythonTestTestAttribute;
  static constexpr const char *pyClassName = "TestAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirPythonTestTestAttributeGetTypeID;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyAiirContext context) {
          return PyTestAttr(context->getRef(), aiirPythonTestTestAttributeGet(
                                                   context.get()->get()));
        },
        nb::arg("context").none() = nb::none());
  }
};
} // namespace python_test
} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir

NB_MODULE(_aiirPythonTestNanobind, m) {
  using namespace aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN;
  m.def(
      "register_python_test_dialect",
      [](DefaultingPyAiirContext context, bool load) {
        AiirDialectHandle pythonTestDialect =
            aiirGetDialectHandle__python_test__();
        aiirDialectHandleRegisterDialect(pythonTestDialect,
                                         context.get()->get());
        if (load) {
          aiirDialectHandleLoadDialect(pythonTestDialect, context.get()->get());
        }
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true);

  m.def(
      "register_dialect",
      [](AiirDialectRegistry registry) {
        AiirDialectHandle pythonTestDialect =
            aiirGetDialectHandle__python_test__();
        aiirDialectHandleInsertDialect(pythonTestDialect, registry);
      },
      nb::arg("registry"),
      // clang-format off
      nb::sig("def register_dialect(registry: " MAKE_AIIR_PYTHON_QUALNAME("ir.DialectRegistry") ") -> None"));
  // clang-format on

  m.def(
      "test_diagnostics_with_errors_and_notes",
      [](DefaultingPyAiirContext ctx) {
        aiir::python::CollectDiagnosticsToStringScope handler(ctx.get()->get());
        aiirPythonTestEmitDiagnosticWithNote(ctx.get()->get());
        throw nb::value_error(handler.takeMessage().c_str());
      },
      nb::arg("context").none() = nb::none());

  using namespace python_test;
  PyTestAttr::bind(m);
  PyTestType::bind(m);
  PyTestIntegerRankedTensorType::bind(m);
  PyTestTensorValue::bind(m);
}
