//===--- DialectNVGPU.cpp - Pybind module for NVGPU dialect API support ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/NVGPU.h"
#include "aiir-c/IR.h"
#include "aiir/Bindings/Python/IRCore.h"
#include "aiir/Bindings/Python/Nanobind.h"
#include "aiir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;
using namespace aiir::python::nanobind_adaptors;

namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {
namespace nvgpu {
struct TensorMapDescriptorType : PyConcreteType<TensorMapDescriptorType> {
  static constexpr IsAFunctionTy isaFunction =
      aiirTypeIsANVGPUTensorMapDescriptorType;
  static constexpr const char *pyClassName = "TensorMapDescriptorType";
  static inline const AiirStringRef name =
      aiirNVGPUTensorMapDescriptorTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const PyType &tensorMemrefType, int swizzle, int l2promo,
           int oobFill, int interleave, DefaultingPyAiirContext context) {
          return TensorMapDescriptorType(
              context->getRef(), aiirNVGPUTensorMapDescriptorTypeGet(
                                     context.get()->get(), tensorMemrefType,
                                     swizzle, l2promo, oobFill, interleave));
        },
        "Gets an instance of TensorMapDescriptorType in the same context",
        nb::arg("tensor_type"), nb::arg("swizzle"), nb::arg("l2promo"),
        nb::arg("oob_fill"), nb::arg("interleave"),
        nb::arg("context").none() = nb::none());
  }
};
} // namespace nvgpu
} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir

NB_MODULE(_aiirDialectsNVGPU, m) {
  m.doc() = "AIIR NVGPU dialect.";

  aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN::nvgpu::TensorMapDescriptorType::
      bind(m);
}
