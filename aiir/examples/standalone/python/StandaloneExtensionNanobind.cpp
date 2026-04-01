//===- StandaloneExtension.cpp - Extension module -------------------------===//
//
// This is the nanobind version of the example module. There is also a pybind11
// example in StandaloneExtensionPybind11.cpp.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone-c/Dialects.h"
#include "aiir-c/Dialect/Arith.h"
#include "aiir/Bindings/Python/IRCore.h"
#include "aiir/Bindings/Python/IRTypes.h"
#include "aiir/Bindings/Python/Nanobind.h"
#include "aiir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

struct PyCustomType
    : aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN::PyConcreteType<PyCustomType> {
  static constexpr IsAFunctionTy isaFunction = aiirStandaloneTypeIsACustomType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirStandaloneCustomTypeGetTypeID;
  static constexpr const char *pyClassName = "CustomType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::string &value,
           aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyAiirContext
               context) {
          return PyCustomType(
              context->getRef(),
              aiirStandaloneCustomTypeGet(
                  context.get()->get(),
                  aiirStringRefCreateFromCString(value.c_str())));
        },
        nb::arg("value"), nb::arg("context").none() = nb::none());
  }
};

NB_MODULE(_standaloneDialectsNanobind, m) {
  //===--------------------------------------------------------------------===//
  // standalone dialect
  //===--------------------------------------------------------------------===//
  auto standaloneM = m.def_submodule("standalone");

  PyCustomType::bind(standaloneM);

  standaloneM.def(
      "register_dialects",
      [](aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyAiirContext
             context,
         bool load) {
        AiirDialectHandle arithHandle = aiirGetDialectHandle__arith__();
        AiirDialectHandle standaloneHandle =
            aiirGetDialectHandle__standalone__();
        AiirContext context_ = context.get()->get();
        aiirDialectHandleRegisterDialect(arithHandle, context_);
        aiirDialectHandleRegisterDialect(standaloneHandle, context_);
        if (load) {
          aiirDialectHandleLoadDialect(arithHandle, context_);
          aiirDialectHandleRegisterDialect(standaloneHandle, context_);
        }
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true);

  standaloneM.def(
      "print_fp_type",
      [](const aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN::PyF16Type &,
         nb::handle stderr_obj) {
        nb::print("this is a fp16 type", /*end=*/nb::handle(), stderr_obj);
      });
  standaloneM.def(
      "print_fp_type",
      [](const aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN::PyF32Type &,
         nb::handle stderr_obj) {
        nb::print("this is a fp32 type", /*end=*/nb::handle(), stderr_obj);
      });
  standaloneM.def(
      "print_fp_type",
      [](const aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN::PyF64Type &,
         nb::handle stderr_obj) {
        nb::print("this is a fp64 type", /*end=*/nb::handle(), stderr_obj);
      });
}
