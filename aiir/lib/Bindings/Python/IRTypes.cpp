//===- IRTypes.cpp - Exports builtin and standard types -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-format off
#include "aiir-c/ExtensibleDialect.h"
#include "aiir/Bindings/Python/IRCore.h"
#include "aiir/Bindings/Python/IRTypes.h"
// clang-format on

#include <optional>
#include <vector>

#include "aiir-c/BuiltinAttributes.h"
#include "aiir-c/BuiltinTypes.h"
#include "aiir-c/Support.h"
#include "aiir/Bindings/Python/NanobindUtils.h"

namespace nb = nanobind;
using namespace aiir;
using namespace aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN;

namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {

int aiirTypeIsAIntegerOrFloat(AiirType type) {
  return aiirTypeIsAInteger(type) || aiirTypeIsABF16(type) ||
         aiirTypeIsAF16(type) || aiirTypeIsAF32(type) || aiirTypeIsAF64(type);
}

void PyIntegerType::bindDerived(ClassTy &c) {
  nb::enum_<Signedness>(c, "Signedness")
      .value("SIGNLESS", Signless)
      .value("SIGNED", Signed)
      .value("UNSIGNED", Unsigned)
      .export_values();

  c.def_static(
      "get_signless",
      [](unsigned width, DefaultingPyAiirContext context) {
        AiirType t = aiirIntegerTypeGet(context->get(), width);
        return PyIntegerType(context->getRef(), t);
      },
      nb::arg("width"), nb::arg("context") = nb::none(),
      "Create a signless integer type");
  c.def_static(
      "get_signed",
      [](unsigned width, DefaultingPyAiirContext context) {
        AiirType t = aiirIntegerTypeSignedGet(context->get(), width);
        return PyIntegerType(context->getRef(), t);
      },
      nb::arg("width"), nb::arg("context") = nb::none(),
      "Create a signed integer type");
  c.def_static(
      "get_unsigned",
      [](unsigned width, DefaultingPyAiirContext context) {
        AiirType t = aiirIntegerTypeUnsignedGet(context->get(), width);
        return PyIntegerType(context->getRef(), t);
      },
      nb::arg("width"), nb::arg("context") = nb::none(),
      "Create an unsigned integer type");
  c.def_static(
      "get",
      [](unsigned width, Signedness signedness,
         DefaultingPyAiirContext context) {
        AiirType t;
        switch (signedness) {
        case Signless:
          t = aiirIntegerTypeGet(context->get(), width);
          break;
        case Signed:
          t = aiirIntegerTypeSignedGet(context->get(), width);
          break;
        case Unsigned:
          t = aiirIntegerTypeUnsignedGet(context->get(), width);
          break;
        }
        return PyIntegerType(context->getRef(), t);
      },
      nb::arg("width"), nb::arg("signedness") = Signless,
      nb::arg("context") = nb::none(), "Create an integer type");
  c.def_prop_ro("signedness", [](PyIntegerType &self) -> Signedness {
    if (aiirIntegerTypeIsSignless(self))
      return Signless;
    if (aiirIntegerTypeIsSigned(self))
      return Signed;
    return Unsigned;
  });
  c.def_prop_ro(
      "width",
      [](PyIntegerType &self) { return aiirIntegerTypeGetWidth(self); },
      "Returns the width of the integer type");
  c.def_prop_ro(
      "is_signless",
      [](PyIntegerType &self) -> bool {
        return aiirIntegerTypeIsSignless(self);
      },
      "Returns whether this is a signless integer");
  c.def_prop_ro(
      "is_signed",
      [](PyIntegerType &self) -> bool { return aiirIntegerTypeIsSigned(self); },
      "Returns whether this is a signed integer");
  c.def_prop_ro(
      "is_unsigned",
      [](PyIntegerType &self) -> bool {
        return aiirIntegerTypeIsUnsigned(self);
      },
      "Returns whether this is an unsigned integer");
}

void PyIndexType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyAiirContext context) {
        AiirType t = aiirIndexTypeGet(context->get());
        return PyIndexType(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a index type.");
}

void PyFloatType::bindDerived(ClassTy &c) {
  c.def_prop_ro(
      "width", [](PyFloatType &self) { return aiirFloatTypeGetWidth(self); },
      "Returns the width of the floating-point type");
}

void PyFloat4E2M1FNType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyAiirContext context) {
        AiirType t = aiirFloat4E2M1FNTypeGet(context->get());
        return PyFloat4E2M1FNType(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a float4_e2m1fn type.");
}

void PyFloat6E2M3FNType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyAiirContext context) {
        AiirType t = aiirFloat6E2M3FNTypeGet(context->get());
        return PyFloat6E2M3FNType(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a float6_e2m3fn type.");
}

void PyFloat6E3M2FNType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyAiirContext context) {
        AiirType t = aiirFloat6E3M2FNTypeGet(context->get());
        return PyFloat6E3M2FNType(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a float6_e3m2fn type.");
}

void PyFloat8E4M3FNType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyAiirContext context) {
        AiirType t = aiirFloat8E4M3FNTypeGet(context->get());
        return PyFloat8E4M3FNType(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a float8_e4m3fn type.");
}

void PyFloat8E5M2Type::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyAiirContext context) {
        AiirType t = aiirFloat8E5M2TypeGet(context->get());
        return PyFloat8E5M2Type(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a float8_e5m2 type.");
}

void PyFloat8E4M3Type::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyAiirContext context) {
        AiirType t = aiirFloat8E4M3TypeGet(context->get());
        return PyFloat8E4M3Type(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a float8_e4m3 type.");
}

void PyFloat8E4M3FNUZType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyAiirContext context) {
        AiirType t = aiirFloat8E4M3FNUZTypeGet(context->get());
        return PyFloat8E4M3FNUZType(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a float8_e4m3fnuz type.");
}

void PyFloat8E4M3B11FNUZType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyAiirContext context) {
        AiirType t = aiirFloat8E4M3B11FNUZTypeGet(context->get());
        return PyFloat8E4M3B11FNUZType(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a float8_e4m3b11fnuz type.");
}

void PyFloat8E5M2FNUZType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyAiirContext context) {
        AiirType t = aiirFloat8E5M2FNUZTypeGet(context->get());
        return PyFloat8E5M2FNUZType(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a float8_e5m2fnuz type.");
}

void PyFloat8E3M4Type::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyAiirContext context) {
        AiirType t = aiirFloat8E3M4TypeGet(context->get());
        return PyFloat8E3M4Type(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a float8_e3m4 type.");
}

void PyFloat8E8M0FNUType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyAiirContext context) {
        AiirType t = aiirFloat8E8M0FNUTypeGet(context->get());
        return PyFloat8E8M0FNUType(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a float8_e8m0fnu type.");
}

void PyBF16Type::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyAiirContext context) {
        AiirType t = aiirBF16TypeGet(context->get());
        return PyBF16Type(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a bf16 type.");
}

void PyF16Type::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyAiirContext context) {
        AiirType t = aiirF16TypeGet(context->get());
        return PyF16Type(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a f16 type.");
}

void PyTF32Type::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyAiirContext context) {
        AiirType t = aiirTF32TypeGet(context->get());
        return PyTF32Type(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a tf32 type.");
}

void PyF32Type::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyAiirContext context) {
        AiirType t = aiirF32TypeGet(context->get());
        return PyF32Type(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a f32 type.");
}

void PyF64Type::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyAiirContext context) {
        AiirType t = aiirF64TypeGet(context->get());
        return PyF64Type(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a f64 type.");
}

void PyNoneType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyAiirContext context) {
        AiirType t = aiirNoneTypeGet(context->get());
        return PyNoneType(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a none type.");
}

void PyComplexType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](PyType &elementType) {
        // The element must be a floating point or integer scalar type.
        if (aiirTypeIsAIntegerOrFloat(elementType)) {
          AiirType t = aiirComplexTypeGet(elementType);
          return PyComplexType(elementType.getContext(), t);
        }
        throw nb::value_error(
            nanobind::detail::join(
                "invalid '",
                nb::cast<std::string>(nb::repr(nb::cast(elementType))),
                "' and expected floating point or integer type.")
                .c_str());
      },
      "Create a complex type");
  c.def_prop_ro(
      "element_type",
      [](PyComplexType &self) -> nb::typed<nb::object, PyType> {
        return PyType(self.getContext(), aiirComplexTypeGetElementType(self))
            .maybeDownCast();
      },
      "Returns element type.");
}

// Shaped Type Interface - ShapedType
void PyShapedType::bindDerived(ClassTy &c) {
  c.def_prop_ro(
      "element_type",
      [](PyShapedType &self) -> nb::typed<nb::object, PyType> {
        return PyType(self.getContext(), aiirShapedTypeGetElementType(self))
            .maybeDownCast();
      },
      "Returns the element type of the shaped type.");
  c.def_prop_ro(
      "has_rank",
      [](PyShapedType &self) -> bool { return aiirShapedTypeHasRank(self); },
      "Returns whether the given shaped type is ranked.");
  c.def_prop_ro(
      "rank",
      [](PyShapedType &self) {
        self.requireHasRank();
        return aiirShapedTypeGetRank(self);
      },
      "Returns the rank of the given ranked shaped type.");
  c.def_prop_ro(
      "has_static_shape",
      [](PyShapedType &self) -> bool {
        return aiirShapedTypeHasStaticShape(self);
      },
      "Returns whether the given shaped type has a static shape.");
  c.def(
      "is_dynamic_dim",
      [](PyShapedType &self, intptr_t dim) -> bool {
        self.requireHasRank();
        return aiirShapedTypeIsDynamicDim(self, dim);
      },
      nb::arg("dim"),
      "Returns whether the dim-th dimension of the given shaped type is "
      "dynamic.");
  c.def(
      "is_static_dim",
      [](PyShapedType &self, intptr_t dim) -> bool {
        self.requireHasRank();
        return aiirShapedTypeIsStaticDim(self, dim);
      },
      nb::arg("dim"),
      "Returns whether the dim-th dimension of the given shaped type is "
      "static.");
  c.def(
      "get_dim_size",
      [](PyShapedType &self, intptr_t dim) {
        self.requireHasRank();
        return aiirShapedTypeGetDimSize(self, dim);
      },
      nb::arg("dim"),
      "Returns the dim-th dimension of the given ranked shaped type.");
  c.def_static(
      "is_dynamic_size",
      [](int64_t size) -> bool { return aiirShapedTypeIsDynamicSize(size); },
      nb::arg("dim_size"),
      "Returns whether the given dimension size indicates a dynamic "
      "dimension.");
  c.def_static(
      "is_static_size",
      [](int64_t size) -> bool { return aiirShapedTypeIsStaticSize(size); },
      nb::arg("dim_size"),
      "Returns whether the given dimension size indicates a static "
      "dimension.");
  c.def(
      "is_dynamic_stride_or_offset",
      [](PyShapedType &self, int64_t val) -> bool {
        self.requireHasRank();
        return aiirShapedTypeIsDynamicStrideOrOffset(val);
      },
      nb::arg("dim_size"),
      "Returns whether the given value is used as a placeholder for dynamic "
      "strides and offsets in shaped types.");
  c.def(
      "is_static_stride_or_offset",
      [](PyShapedType &self, int64_t val) -> bool {
        self.requireHasRank();
        return aiirShapedTypeIsStaticStrideOrOffset(val);
      },
      nb::arg("dim_size"),
      "Returns whether the given shaped type stride or offset value is "
      "statically-sized.");
  c.def_prop_ro(
      "shape",
      [](PyShapedType &self) {
        self.requireHasRank();

        std::vector<int64_t> shape;
        int64_t rank = aiirShapedTypeGetRank(self);
        shape.reserve(rank);
        for (int64_t i = 0; i < rank; ++i)
          shape.push_back(aiirShapedTypeGetDimSize(self, i));
        return shape;
      },
      "Returns the shape of the ranked shaped type as a list of integers.");
  c.def_static(
      "get_dynamic_size", []() { return aiirShapedTypeGetDynamicSize(); },
      "Returns the value used to indicate dynamic dimensions in shaped "
      "types.");
  c.def_static(
      "get_dynamic_stride_or_offset",
      []() { return aiirShapedTypeGetDynamicStrideOrOffset(); },
      "Returns the value used to indicate dynamic strides or offsets in "
      "shaped types.");
}

void PyShapedType::requireHasRank() {
  if (!aiirShapedTypeHasRank(*this)) {
    throw nb::value_error(
        "calling this method requires that the type has a rank.");
  }
}

const PyShapedType::IsAFunctionTy PyShapedType::isaFunction = aiirTypeIsAShaped;

void PyVectorType::bindDerived(ClassTy &c) {
  c.def_static("get", &PyVectorType::getChecked, nb::arg("shape"),
               nb::arg("element_type"), nb::kw_only(),
               nb::arg("scalable") = nb::none(),
               nb::arg("scalable_dims") = nb::none(),
               nb::arg("loc") = nb::none(), "Create a vector type")
      .def_static("get_unchecked", &PyVectorType::get, nb::arg("shape"),
                  nb::arg("element_type"), nb::kw_only(),
                  nb::arg("scalable") = nb::none(),
                  nb::arg("scalable_dims") = nb::none(),
                  nb::arg("context") = nb::none(), "Create a vector type")
      .def_prop_ro("scalable",
                   [](PyType self) { return aiirVectorTypeIsScalable(self); })
      .def_prop_ro("scalable_dims", [](PyType self) {
        std::vector<bool> scalableDims;
        size_t rank = static_cast<size_t>(aiirShapedTypeGetRank(self));
        scalableDims.reserve(rank);
        for (size_t i = 0; i < rank; ++i)
          scalableDims.push_back(aiirVectorTypeIsDimScalable(self, i));
        return scalableDims;
      });
}

PyVectorType
PyVectorType::getChecked(std::vector<int64_t> shape, PyType &elementType,
                         std::optional<nb::sequence> scalable,
                         std::optional<std::vector<int64_t>> scalableDims,
                         DefaultingPyLocation loc) {
  if (scalable && scalableDims) {
    throw nb::value_error("'scalable' and 'scalable_dims' kwargs "
                          "are mutually exclusive.");
  }

  PyAiirContext::ErrorCapture errors(loc->getContext());
  AiirType type;
  if (scalable) {
    if (nb::len(*scalable) != shape.size())
      throw nb::value_error("Expected len(scalable) == len(shape).");

    std::vector<char> scalableDimFlags;
    scalableDimFlags.reserve(nb::len(*scalable));
    for (const nb::handle &h : *scalable) {
      scalableDimFlags.push_back(nb::cast<bool>(h) ? 1 : 0);
    }
    type = aiirVectorTypeGetScalableChecked(
        loc, shape.size(), shape.data(),
        reinterpret_cast<const bool *>(scalableDimFlags.data()), elementType);
  } else if (scalableDims) {
    std::vector<char> scalableDimFlags(shape.size(), 0);
    for (int64_t dim : *scalableDims) {
      if (static_cast<size_t>(dim) >= scalableDimFlags.size() || dim < 0)
        throw nb::value_error("Scalable dimension index out of bounds.");
      scalableDimFlags[dim] = 1;
    }
    type = aiirVectorTypeGetScalableChecked(
        loc, shape.size(), shape.data(),
        reinterpret_cast<const bool *>(scalableDimFlags.data()), elementType);
  } else {
    type =
        aiirVectorTypeGetChecked(loc, shape.size(), shape.data(), elementType);
  }
  if (aiirTypeIsNull(type))
    throw AIIRError("Invalid type", errors.take());
  return PyVectorType(elementType.getContext(), type);
}

PyVectorType PyVectorType::get(std::vector<int64_t> shape, PyType &elementType,
                               std::optional<nb::sequence> scalable,
                               std::optional<std::vector<int64_t>> scalableDims,
                               DefaultingPyAiirContext context) {
  if (scalable && scalableDims) {
    throw nb::value_error("'scalable' and 'scalable_dims' kwargs "
                          "are mutually exclusive.");
  }

  PyAiirContext::ErrorCapture errors(context->getRef());
  AiirType type;
  if (scalable) {
    if (nb::len(*scalable) != shape.size())
      throw nb::value_error("Expected len(scalable) == len(shape).");

    std::vector<char> scalableDimFlags;
    scalableDimFlags.reserve(nb::len(*scalable));
    for (const nb::handle &h : *scalable) {
      scalableDimFlags.push_back(nb::cast<bool>(h) ? 1 : 0);
    }
    type = aiirVectorTypeGetScalable(
        shape.size(), shape.data(),
        reinterpret_cast<const bool *>(scalableDimFlags.data()), elementType);
  } else if (scalableDims) {
    std::vector<char> scalableDimFlags(shape.size(), 0);
    for (int64_t dim : *scalableDims) {
      if (static_cast<size_t>(dim) >= scalableDimFlags.size() || dim < 0)
        throw nb::value_error("Scalable dimension index out of bounds.");
      scalableDimFlags[dim] = 1;
    }
    type = aiirVectorTypeGetScalable(
        shape.size(), shape.data(),
        reinterpret_cast<const bool *>(scalableDimFlags.data()), elementType);
  } else {
    type = aiirVectorTypeGet(shape.size(), shape.data(), elementType);
  }
  if (aiirTypeIsNull(type))
    throw AIIRError("Invalid type", errors.take());
  return PyVectorType(elementType.getContext(), type);
}

void PyRankedTensorType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](std::vector<int64_t> shape, PyType &elementType,
         std::optional<PyAttribute> &encodingAttr, DefaultingPyLocation loc) {
        PyAiirContext::ErrorCapture errors(loc->getContext());
        AiirType t = aiirRankedTensorTypeGetChecked(
            loc, shape.size(), shape.data(), elementType,
            encodingAttr ? encodingAttr->get() : aiirAttributeGetNull());
        if (aiirTypeIsNull(t))
          throw AIIRError("Invalid type", errors.take());
        return PyRankedTensorType(elementType.getContext(), t);
      },
      nb::arg("shape"), nb::arg("element_type"),
      nb::arg("encoding") = nb::none(), nb::arg("loc") = nb::none(),
      "Create a ranked tensor type");
  c.def_static(
      "get_unchecked",
      [](std::vector<int64_t> shape, PyType &elementType,
         std::optional<PyAttribute> &encodingAttr,
         DefaultingPyAiirContext context) {
        PyAiirContext::ErrorCapture errors(context->getRef());
        AiirType t = aiirRankedTensorTypeGet(
            shape.size(), shape.data(), elementType,
            encodingAttr ? encodingAttr->get() : aiirAttributeGetNull());
        if (aiirTypeIsNull(t))
          throw AIIRError("Invalid type", errors.take());
        return PyRankedTensorType(elementType.getContext(), t);
      },
      nb::arg("shape"), nb::arg("element_type"),
      nb::arg("encoding") = nb::none(), nb::arg("context") = nb::none(),
      "Create a ranked tensor type");
  c.def_prop_ro(
      "encoding",
      [](PyRankedTensorType &self)
          -> std::optional<nb::typed<nb::object, PyAttribute>> {
        AiirAttribute encoding = aiirRankedTensorTypeGetEncoding(self.get());
        if (aiirAttributeIsNull(encoding))
          return std::nullopt;
        return PyAttribute(self.getContext(), encoding).maybeDownCast();
      });
}

void PyUnrankedTensorType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](PyType &elementType, DefaultingPyLocation loc) {
        PyAiirContext::ErrorCapture errors(loc->getContext());
        AiirType t = aiirUnrankedTensorTypeGetChecked(loc, elementType);
        if (aiirTypeIsNull(t))
          throw AIIRError("Invalid type", errors.take());
        return PyUnrankedTensorType(elementType.getContext(), t);
      },
      nb::arg("element_type"), nb::arg("loc") = nb::none(),
      "Create a unranked tensor type");
  c.def_static(
      "get_unchecked",
      [](PyType &elementType, DefaultingPyAiirContext context) {
        PyAiirContext::ErrorCapture errors(context->getRef());
        AiirType t = aiirUnrankedTensorTypeGet(elementType);
        if (aiirTypeIsNull(t))
          throw AIIRError("Invalid type", errors.take());
        return PyUnrankedTensorType(elementType.getContext(), t);
      },
      nb::arg("element_type"), nb::arg("context") = nb::none(),
      "Create a unranked tensor type");
}

void PyMemRefType::bindDerived(ClassTy &c) {
  c.def_static(
       "get",
       [](std::vector<int64_t> shape, PyType &elementType, PyAttribute *layout,
          PyAttribute *memorySpace, DefaultingPyLocation loc) {
         PyAiirContext::ErrorCapture errors(loc->getContext());
         AiirAttribute layoutAttr = layout ? *layout : aiirAttributeGetNull();
         AiirAttribute memSpaceAttr =
             memorySpace ? *memorySpace : aiirAttributeGetNull();
         AiirType t =
             aiirMemRefTypeGetChecked(loc, elementType, shape.size(),
                                      shape.data(), layoutAttr, memSpaceAttr);
         if (aiirTypeIsNull(t))
           throw AIIRError("Invalid type", errors.take());
         return PyMemRefType(elementType.getContext(), t);
       },
       nb::arg("shape"), nb::arg("element_type"),
       nb::arg("layout") = nb::none(), nb::arg("memory_space") = nb::none(),
       nb::arg("loc") = nb::none(), "Create a memref type")
      .def_static(
          "get_unchecked",
          [](std::vector<int64_t> shape, PyType &elementType,
             PyAttribute *layout, PyAttribute *memorySpace,
             DefaultingPyAiirContext context) {
            PyAiirContext::ErrorCapture errors(context->getRef());
            AiirAttribute layoutAttr =
                layout ? *layout : aiirAttributeGetNull();
            AiirAttribute memSpaceAttr =
                memorySpace ? *memorySpace : aiirAttributeGetNull();
            AiirType t =
                aiirMemRefTypeGet(elementType, shape.size(), shape.data(),
                                  layoutAttr, memSpaceAttr);
            if (aiirTypeIsNull(t))
              throw AIIRError("Invalid type", errors.take());
            return PyMemRefType(elementType.getContext(), t);
          },
          nb::arg("shape"), nb::arg("element_type"),
          nb::arg("layout") = nb::none(), nb::arg("memory_space") = nb::none(),
          nb::arg("context") = nb::none(), "Create a memref type")
      .def_prop_ro(
          "layout",
          [](PyMemRefType &self) -> nb::typed<nb::object, PyAttribute> {
            return PyAttribute(self.getContext(), aiirMemRefTypeGetLayout(self))
                .maybeDownCast();
          },
          "The layout of the MemRef type.")
      .def(
          "get_strides_and_offset",
          [](PyMemRefType &self) -> std::pair<std::vector<int64_t>, int64_t> {
            std::vector<int64_t> strides(aiirShapedTypeGetRank(self));
            int64_t offset;
            if (aiirLogicalResultIsFailure(aiirMemRefTypeGetStridesAndOffset(
                    self, strides.data(), &offset)))
              throw std::runtime_error(
                  "Failed to extract strides and offset from memref.");
            return {strides, offset};
          },
          "The strides and offset of the MemRef type.")
      .def_prop_ro(
          "affine_map",
          [](PyMemRefType &self) -> PyAffineMap {
            AiirAffineMap map = aiirMemRefTypeGetAffineMap(self);
            return PyAffineMap(self.getContext(), map);
          },
          "The layout of the MemRef type as an affine map.")
      .def_prop_ro(
          "memory_space",
          [](PyMemRefType &self)
              -> std::optional<nb::typed<nb::object, PyAttribute>> {
            AiirAttribute a = aiirMemRefTypeGetMemorySpace(self);
            if (aiirAttributeIsNull(a))
              return std::nullopt;
            return PyAttribute(self.getContext(), a).maybeDownCast();
          },
          "Returns the memory space of the given MemRef type.");
}

void PyUnrankedMemRefType::bindDerived(ClassTy &c) {
  c.def_static(
       "get",
       [](PyType &elementType, PyAttribute *memorySpace,
          DefaultingPyLocation loc) {
         PyAiirContext::ErrorCapture errors(loc->getContext());
         AiirAttribute memSpaceAttr = {};
         if (memorySpace)
           memSpaceAttr = *memorySpace;

         AiirType t =
             aiirUnrankedMemRefTypeGetChecked(loc, elementType, memSpaceAttr);
         if (aiirTypeIsNull(t))
           throw AIIRError("Invalid type", errors.take());
         return PyUnrankedMemRefType(elementType.getContext(), t);
       },
       nb::arg("element_type"), nb::arg("memory_space").none(),
       nb::arg("loc") = nb::none(), "Create a unranked memref type")
      .def_static(
          "get_unchecked",
          [](PyType &elementType, PyAttribute *memorySpace,
             DefaultingPyAiirContext context) {
            PyAiirContext::ErrorCapture errors(context->getRef());
            AiirAttribute memSpaceAttr = {};
            if (memorySpace)
              memSpaceAttr = *memorySpace;

            AiirType t = aiirUnrankedMemRefTypeGet(elementType, memSpaceAttr);
            if (aiirTypeIsNull(t))
              throw AIIRError("Invalid type", errors.take());
            return PyUnrankedMemRefType(elementType.getContext(), t);
          },
          nb::arg("element_type"), nb::arg("memory_space").none(),
          nb::arg("context") = nb::none(), "Create a unranked memref type")
      .def_prop_ro(
          "memory_space",
          [](PyUnrankedMemRefType &self)
              -> std::optional<nb::typed<nb::object, PyAttribute>> {
            AiirAttribute a = aiirUnrankedMemrefGetMemorySpace(self);
            if (aiirAttributeIsNull(a))
              return std::nullopt;
            return PyAttribute(self.getContext(), a).maybeDownCast();
          },
          "Returns the memory space of the given Unranked MemRef type.");
}

void PyTupleType::bindDerived(ClassTy &c) {
  c.def_static(
      "get_tuple",
      [](const std::vector<PyType> &elements, DefaultingPyAiirContext context) {
        std::vector<AiirType> aiirElements;
        aiirElements.reserve(elements.size());
        for (const auto &element : elements)
          aiirElements.push_back(element.get());
        AiirType t = aiirTupleTypeGet(context->get(), elements.size(),
                                      aiirElements.data());
        return PyTupleType(context->getRef(), t);
      },
      nb::arg("elements"), nb::arg("context") = nb::none(),
      "Create a tuple type");
  c.def(
      "get_type",
      [](PyTupleType &self, intptr_t pos) -> nb::typed<nb::object, PyType> {
        return PyType(self.getContext(), aiirTupleTypeGetType(self, pos))
            .maybeDownCast();
      },
      nb::arg("pos"), "Returns the pos-th type in the tuple type.");
  c.def_prop_ro(
      "num_types",
      [](PyTupleType &self) -> intptr_t {
        return aiirTupleTypeGetNumTypes(self);
      },
      "Returns the number of types contained in a tuple.");
}

void PyFunctionType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](std::vector<PyType> inputs, std::vector<PyType> results,
         DefaultingPyAiirContext context) {
        std::vector<AiirType> aiirInputs;
        aiirInputs.reserve(inputs.size());
        for (const auto &input : inputs)
          aiirInputs.push_back(input.get());
        std::vector<AiirType> aiirResults;
        aiirResults.reserve(results.size());
        for (const auto &result : results)
          aiirResults.push_back(result.get());

        AiirType t = aiirFunctionTypeGet(context->get(), inputs.size(),
                                         aiirInputs.data(), results.size(),
                                         aiirResults.data());
        return PyFunctionType(context->getRef(), t);
      },
      nb::arg("inputs"), nb::arg("results"), nb::arg("context") = nb::none(),
      "Gets a FunctionType from a list of input and result types");
  c.def_prop_ro(
      "inputs",
      [](PyFunctionType &self) -> nb::typed<nb::list, PyType> {
        AiirType t = self;
        nb::list types;
        for (intptr_t i = 0, e = aiirFunctionTypeGetNumInputs(self); i < e;
             ++i) {
          types.append(aiirFunctionTypeGetInput(t, i));
        }
        return types;
      },
      "Returns the list of input types in the FunctionType.");
  c.def_prop_ro(
      "results",
      [](PyFunctionType &self) -> nb::typed<nb::list, PyType> {
        nb::list types;
        for (intptr_t i = 0, e = aiirFunctionTypeGetNumResults(self); i < e;
             ++i) {
          types.append(aiirFunctionTypeGetResult(self, i));
        }
        return types;
      },
      "Returns the list of result types in the FunctionType.");
}

void PyOpaqueType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](const std::string &dialectNamespace, const std::string &typeData,
         DefaultingPyAiirContext context) {
        AiirType type =
            aiirOpaqueTypeGet(context->get(), toAiirStringRef(dialectNamespace),
                              toAiirStringRef(typeData));
        return PyOpaqueType(context->getRef(), type);
      },
      nb::arg("dialect_namespace"), nb::arg("buffer"),
      nb::arg("context") = nb::none(),
      "Create an unregistered (opaque) dialect type.");
  c.def_prop_ro(
      "dialect_namespace",
      [](PyOpaqueType &self) {
        AiirStringRef stringRef = aiirOpaqueTypeGetDialectNamespace(self);
        return nb::str(stringRef.data, stringRef.length);
      },
      "Returns the dialect namespace for the Opaque type as a string.");
  c.def_prop_ro(
      "data",
      [](PyOpaqueType &self) {
        AiirStringRef stringRef = aiirOpaqueTypeGetData(self);
        return nb::str(stringRef.data, stringRef.length);
      },
      "Returns the data for the Opaque type as a string.");
}

static AiirDynamicTypeDefinition
getDynamicTypeDef(const std::string &fullTypeName,
                  DefaultingPyAiirContext context) {
  size_t dotPos = fullTypeName.find('.');
  if (dotPos == std::string::npos) {
    throw nb::value_error("Expected full type name to be in the format "
                          "'<dialectName>.<typeName>'.");
  }

  std::string dialectName = fullTypeName.substr(0, dotPos);
  std::string typeName = fullTypeName.substr(dotPos + 1);
  PyDialects dialects(context->getRef());
  AiirDialect dialect = dialects.getDialectForKey(dialectName, false);
  if (!aiirDialectIsAExtensibleDialect(dialect))
    throw nb::value_error(
        ("Dialect '" + dialectName + "' is not an extensible dialect.")
            .c_str());

  AiirDynamicTypeDefinition typeDef = aiirExtensibleDialectLookupTypeDefinition(
      dialect, toAiirStringRef(typeName));
  if (typeDef.ptr == nullptr) {
    throw nb::value_error(("Dialect '" + dialectName +
                           "' does not contain a type named '" + typeName +
                           "'.")
                              .c_str());
  }

  return typeDef;
}

void PyDynamicType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](const std::string &fullTypeName, const std::vector<PyAttribute> &attrs,
         DefaultingPyAiirContext context) {
        AiirDynamicTypeDefinition typeDef =
            getDynamicTypeDef(fullTypeName, context);

        std::vector<AiirAttribute> aiirAttrs;
        aiirAttrs.reserve(attrs.size());
        for (const auto &attr : attrs)
          aiirAttrs.push_back(attr.get());
        AiirType t =
            aiirDynamicTypeGet(typeDef, aiirAttrs.data(), aiirAttrs.size());
        return PyDynamicType(context->getRef(), t);
      },
      nb::arg("full_type_name"), nb::arg("attributes"),
      nb::arg("context") = nb::none(), "Create a dynamic type.");
  c.def_prop_ro(
      "params",
      [](PyDynamicType &self) {
        size_t numParams = aiirDynamicTypeGetNumParams(self);
        std::vector<PyAttribute> params;
        params.reserve(numParams);
        for (size_t i = 0; i < numParams; ++i)
          params.emplace_back(self.getContext(),
                              aiirDynamicTypeGetParam(self, i));
        return params;
      },
      "Returns the parameters of the dynamic type as a list of attributes.");
  c.def_prop_ro("type_name", [](PyDynamicType &self) {
    AiirDynamicTypeDefinition typeDef = aiirDynamicTypeGetTypeDef(self);
    AiirStringRef name = aiirDynamicTypeDefinitionGetName(typeDef);
    AiirDialect dialect = aiirDynamicTypeDefinitionGetDialect(typeDef);
    AiirStringRef dialectNamespace = aiirDialectGetNamespace(dialect);
    return std::string(dialectNamespace.data, dialectNamespace.length) + "." +
           std::string(name.data, name.length);
  });
  c.def_static(
      "lookup_typeid",
      [](const std::string &fullTypeName, DefaultingPyAiirContext context) {
        AiirDynamicTypeDefinition typeDef =
            getDynamicTypeDef(fullTypeName, context);
        return PyTypeID(aiirDynamicTypeDefinitionGetTypeID(typeDef));
      },
      nb::arg("full_type_name"), nb::arg("context") = nb::none(),
      "Look up the TypeID for the given dynamic type name.");
}

void populateIRTypes(nb::module_ &m) {
  PyIntegerType::bind(m);
  PyFloatType::bind(m);
  PyIndexType::bind(m);
  PyFloat4E2M1FNType::bind(m);
  PyFloat6E2M3FNType::bind(m);
  PyFloat6E3M2FNType::bind(m);
  PyFloat8E4M3FNType::bind(m);
  PyFloat8E5M2Type::bind(m);
  PyFloat8E4M3Type::bind(m);
  PyFloat8E4M3FNUZType::bind(m);
  PyFloat8E4M3B11FNUZType::bind(m);
  PyFloat8E5M2FNUZType::bind(m);
  PyFloat8E3M4Type::bind(m);
  PyFloat8E8M0FNUType::bind(m);
  PyBF16Type::bind(m);
  PyF16Type::bind(m);
  PyTF32Type::bind(m);
  PyF32Type::bind(m);
  PyF64Type::bind(m);
  PyNoneType::bind(m);
  PyComplexType::bind(m);
  PyShapedType::bind(m);
  PyVectorType::bind(m);
  PyRankedTensorType::bind(m);
  PyUnrankedTensorType::bind(m);
  PyMemRefType::bind(m);
  PyUnrankedMemRefType::bind(m);
  PyTupleType::bind(m);
  PyFunctionType::bind(m);
  PyOpaqueType::bind(m);
  PyDynamicType::bind(m);
}
} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir
