//===- DialectGPU.cpp - Pybind module for the GPU passes ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "aiir-c/Dialect/GPU.h"
#include "aiir-c/IR.h"
#include "aiir-c/Support.h"
#include "aiir/Bindings/Python/IRAttributes.h"
#include "aiir/Bindings/Python/IRCore.h"
#include "aiir/Bindings/Python/Nanobind.h"
#include "aiir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;
using namespace nanobind::literals;
using namespace aiir::python::nanobind_adaptors;

namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {
namespace gpu {
// -----------------------------------------------------------------------------
// AsyncTokenType
// -----------------------------------------------------------------------------

struct AsyncTokenType : PyConcreteType<AsyncTokenType> {
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAGPUAsyncTokenType;
  static constexpr const char *pyClassName = "AsyncTokenType";
  static inline const AiirStringRef name = aiirGPUAsyncTokenTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyAiirContext context) {
          return AsyncTokenType(context->getRef(),
                                aiirGPUAsyncTokenTypeGet(context.get()->get()));
        },
        "Gets an instance of AsyncTokenType in the same context",
        nb::arg("context").none() = nb::none());
  }
};

//===-------------------------------------------------------------------===//
// ObjectAttr
//===-------------------------------------------------------------------===//

struct ObjectAttr : PyConcreteAttribute<ObjectAttr> {
  static constexpr IsAFunctionTy isaFunction = aiirAttributeIsAGPUObjectAttr;
  static constexpr const char *pyClassName = "ObjectAttr";
  static inline const AiirStringRef name = aiirGPUObjectAttrGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const PyAttribute &target, uint32_t format, const nb::bytes &object,
           std::optional<PyDictAttribute> aiirObjectProps,
           std::optional<PyAttribute> aiirKernelsAttr,
           DefaultingPyAiirContext context) {
          AiirStringRef objectStrRef = aiirStringRefCreate(
              static_cast<char *>(const_cast<void *>(object.data())),
              object.size());
          return ObjectAttr(
              context->getRef(),
              aiirGPUObjectAttrGetWithKernels(
                  aiirAttributeGetContext(target), target, format, objectStrRef,
                  aiirObjectProps.has_value() ? *aiirObjectProps
                                              : AiirAttribute{nullptr},
                  aiirKernelsAttr.has_value() ? *aiirKernelsAttr
                                              : AiirAttribute{nullptr}));
        },
        "target"_a, "format"_a, "object"_a, "properties"_a = nb::none(),
        "kernels"_a = nb::none(), "context"_a = nb::none(),
        "Gets a gpu.object from parameters.");

    c.def_prop_ro("target", [](ObjectAttr &self) {
      return PyAttribute(self.getContext(), aiirGPUObjectAttrGetTarget(self))
          .maybeDownCast();
    });
    c.def_prop_ro("format", [](const ObjectAttr &self) {
      return aiirGPUObjectAttrGetFormat(self);
    });
    c.def_prop_ro("object", [](const ObjectAttr &self) {
      AiirStringRef stringRef = aiirGPUObjectAttrGetObject(self);
      return nb::bytes(stringRef.data, stringRef.length);
    });
    c.def_prop_ro(
        "properties", [](ObjectAttr &self) -> std::optional<PyDictAttribute> {
          if (aiirGPUObjectAttrHasProperties(self))
            return PyDictAttribute(self.getContext(),
                                   aiirGPUObjectAttrGetProperties(self));
          return std::nullopt;
        });
    c.def_prop_ro("kernels",
                  [](ObjectAttr &self)
                      -> std::optional<nb::typed<nb::object, PyAttribute>> {
                    if (aiirGPUObjectAttrHasKernels(self))
                      return PyAttribute(self.getContext(),
                                         aiirGPUObjectAttrGetKernels(self))
                          .maybeDownCast();
                    return std::nullopt;
                  });
  }
};
} // namespace gpu
} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

NB_MODULE(_aiirDialectsGPU, m) {
  m.doc() = "AIIR GPU Dialect";

  aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN::gpu::AsyncTokenType::bind(m);
  aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN::gpu::ObjectAttr::bind(m);
}
