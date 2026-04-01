//===--- DialectAMDGPU.cpp - Pybind module for AMDGPU dialect API support -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/AMDGPU.h"
#include "aiir-c/IR.h"
#include "aiir/Bindings/Python/IRCore.h"
#include "aiir/Bindings/Python/Nanobind.h"
#include "aiir/Bindings/Python/NanobindAdaptors.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;
using namespace aiir::python::nanobind_adaptors;

namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {
namespace amdgpu {
struct TDMBaseType : PyConcreteType<TDMBaseType> {
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAAMDGPUTDMBaseType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirAMDGPUTDMBaseTypeGetTypeID;
  static constexpr const char *pyClassName = "TDMBaseType";
  static inline const AiirStringRef name = aiirAMDGPUTDMBaseTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const PyType &elementType, DefaultingPyAiirContext context) {
          return TDMBaseType(
              context->getRef(),
              aiirAMDGPUTDMBaseTypeGet(context.get()->get(), elementType));
        },
        "Gets an instance of TDMBaseType in the same context",
        nb::arg("element_type"), nb::arg("context").none() = nb::none());
  }
};

struct TDMDescriptorType : PyConcreteType<TDMDescriptorType> {
  static constexpr IsAFunctionTy isaFunction =
      aiirTypeIsAAMDGPUTDMDescriptorType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirAMDGPUTDMDescriptorTypeGetTypeID;
  static constexpr const char *pyClassName = "TDMDescriptorType";
  static inline const AiirStringRef name = aiirAMDGPUTDMDescriptorTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyAiirContext context) {
          return TDMDescriptorType(
              context->getRef(),
              aiirAMDGPUTDMDescriptorTypeGet(context.get()->get()));
        },
        "Gets an instance of TDMDescriptorType in the same context",
        nb::arg("context").none() = nb::none());
  }
};

struct TDMGatherBaseType : PyConcreteType<TDMGatherBaseType> {
  static constexpr IsAFunctionTy isaFunction =
      aiirTypeIsAAMDGPUTDMGatherBaseType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirAMDGPUTDMGatherBaseTypeGetTypeID;
  static constexpr const char *pyClassName = "TDMGatherBaseType";
  static inline const AiirStringRef name = aiirAMDGPUTDMGatherBaseTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const PyType &elementType, const PyType &indexType,
           DefaultingPyAiirContext context) {
          return TDMGatherBaseType(
              context->getRef(),
              aiirAMDGPUTDMGatherBaseTypeGet(context.get()->get(), elementType,
                                             indexType));
        },
        "Gets an instance of TDMGatherBaseType in the same context",
        nb::arg("element_type"), nb::arg("index_type"),
        nb::arg("context").none() = nb::none());
  }
};

static void populateDialectAMDGPUSubmodule(nb::module_ &m) {
  TDMBaseType::bind(m);
  TDMDescriptorType::bind(m);
  TDMGatherBaseType::bind(m);
}
} // namespace amdgpu
} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir

NB_MODULE(_aiirDialectsAMDGPU, m) {
  m.doc() = "AIIR AMDGPU dialect.";

  aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN::amdgpu::
      populateDialectAMDGPUSubmodule(m);
}
