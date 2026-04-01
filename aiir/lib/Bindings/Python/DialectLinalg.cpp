//===- DialectLinalg.cpp - Nanobind module for Linalg dialect API support -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/Linalg.h"
#include "aiir-c/IR.h"
#include "aiir/Bindings/Python/IRAttributes.h"
#include "aiir/Bindings/Python/IRCore.h"
#include "aiir/Bindings/Python/Nanobind.h"
#include "aiir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;
using namespace aiir::python::nanobind_adaptors;
namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {
namespace linalg {

struct PyLinalgContractionDimensions : AiirLinalgContractionDimensions {
  PyLinalgContractionDimensions(const AiirLinalgContractionDimensions &dims) {
    batch = dims.batch;
    m = dims.m;
    n = dims.n;
    k = dims.k;
  }
};

struct PyLinalgConvolutionDimensions : AiirLinalgConvolutionDimensions {
  PyLinalgConvolutionDimensions(const AiirLinalgConvolutionDimensions &dims) {
    batch = dims.batch;
    outputImage = dims.outputImage;
    outputChannel = dims.outputChannel;
    filterLoop = dims.filterLoop;
    inputChannel = dims.inputChannel;
    depth = dims.depth;
    strides = dims.strides;
    dilations = dims.dilations;
  }
};

static std::optional<PyLinalgContractionDimensions>
InferContractionDimensions(PyOperationBase &op) {
  AiirLinalgContractionDimensions dims =
      aiirLinalgInferContractionDimensions(op.getOperation());

  // Detect "empty" result. This occurs when `op` is not a contraction op,
  // or when `linalg::inferContractionDims` fails.
  if (aiirAttributeIsNull(dims.batch) && aiirAttributeIsNull(dims.m) &&
      aiirAttributeIsNull(dims.n) && aiirAttributeIsNull(dims.k)) {
    return std::nullopt;
  }
  return dims;
}

static std::optional<PyLinalgConvolutionDimensions>
InferConvolutionDimensions(PyOperationBase &op) {
  AiirLinalgConvolutionDimensions dims =
      aiirLinalgInferConvolutionDimensions(op.getOperation());

  // Detect "empty" result. This occurs when `op` is not a convolution op,
  // or when `linalg::inferConvolutionDims` fails.
  if (aiirAttributeIsNull(dims.batch) &&
      aiirAttributeIsNull(dims.outputImage) &&
      aiirAttributeIsNull(dims.outputChannel) &&
      aiirAttributeIsNull(dims.filterLoop) &&
      aiirAttributeIsNull(dims.inputChannel) &&
      aiirAttributeIsNull(dims.depth) && aiirAttributeIsNull(dims.strides) &&
      aiirAttributeIsNull(dims.dilations)) {
    return std::nullopt;
  }

  return dims;
}

static void populateDialectLinalgSubmodule(nb::module_ m) {
  m.def(
      "fill_builtin_region",
      [](PyOperationBase &op) {
        aiirLinalgFillBuiltinNamedOpRegion(op.getOperation());
      },
      nb::arg("op"),
      "Fill the region for `op`, which is assumed to be a builtin named Linalg "
      "op.");

  m.def(
      "isa_contraction_op",
      [](PyOperationBase &op) {
        return aiirLinalgIsAContractionOp(op.getOperation());
      },
      "Checks if the given operation is a Linalg contraction operation.",
      nb::arg("op"));

  nb::class_<PyLinalgContractionDimensions>(m, "ContractionDimensions")
      .def_prop_ro(
          "batch",
          [](const PyLinalgContractionDimensions &self) { return self.batch; })
      .def_prop_ro(
          "m", [](const PyLinalgContractionDimensions &self) { return self.m; })
      .def_prop_ro(
          "n", [](const PyLinalgContractionDimensions &self) { return self.n; })
      .def_prop_ro("k", [](const PyLinalgContractionDimensions &self) {
        return self.k;
      });

  m.def("infer_contraction_dimensions", &InferContractionDimensions,
        "Infers contraction dimensions (batch/m/n/k) for a Linalg contraction "
        "op.",
        nb::arg("op"));

  m.def(
      "infer_contraction_dimensions_from_maps",
      [](std::vector<PyAffineMap> indexingMaps)
          -> std::optional<PyLinalgContractionDimensions> {
        if (indexingMaps.empty())
          return std::nullopt;

        std::vector<AiirAffineMap> indexingMaps_(indexingMaps.size());
        std::copy(indexingMaps.begin(), indexingMaps.end(),
                  indexingMaps_.begin());
        AiirLinalgContractionDimensions dims =
            aiirLinalgInferContractionDimensionsFromMaps(indexingMaps_.data(),
                                                         indexingMaps_.size());

        // Detect "empty" result from invalid input or failed inference.
        if (aiirAttributeIsNull(dims.batch) && aiirAttributeIsNull(dims.m) &&
            aiirAttributeIsNull(dims.n) && aiirAttributeIsNull(dims.k)) {
          return std::nullopt;
        }
        return dims;
      },
      "Infers contraction dimensions (batch/m/n/k) from a list of affine "
      "maps.",
      nb::arg("indexing_maps"));

  m.def(
      "isa_convolution_op",
      [](PyOperationBase &op) {
        return aiirLinalgIsAConvolutionOp(op.getOperation());
      },
      "Checks if the given operation is a Linalg convolution operation.",
      nb::arg("op"));

  nb::class_<PyLinalgConvolutionDimensions>(m, "ConvolutionDimensions")
      .def_prop_ro(
          "batch",
          [](const PyLinalgConvolutionDimensions &self) { return self.batch; })
      .def_prop_ro("output_image",
                   [](const PyLinalgConvolutionDimensions &self) {
                     return self.outputImage;
                   })
      .def_prop_ro("output_channel",
                   [](const PyLinalgConvolutionDimensions &self) {
                     return self.outputChannel;
                   })
      .def_prop_ro("filter_loop",
                   [](const PyLinalgConvolutionDimensions &self) {
                     return self.filterLoop;
                   })
      .def_prop_ro("input_channel",
                   [](const PyLinalgConvolutionDimensions &self) {
                     return self.inputChannel;
                   })
      .def_prop_ro(
          "depth",
          [](const PyLinalgConvolutionDimensions &self) { return self.depth; })
      .def_prop_ro("strides",
                   [](const PyLinalgConvolutionDimensions &self) {
                     return self.strides;
                   })
      .def_prop_ro("dilations", [](const PyLinalgConvolutionDimensions &self) {
        return self.dilations;
      });

  m.def("infer_convolution_dimensions", &InferConvolutionDimensions,
        "Infers convolution dimensions", nb::arg("op"));

  m.def(
      "get_indexing_maps",
      [](PyOperationBase &op) -> std::optional<PyArrayAttribute> {
        AiirAttribute attr =
            aiirLinalgGetIndexingMapsAttribute(op.getOperation());
        if (aiirAttributeIsNull(attr))
          return std::nullopt;
        return PyArrayAttribute(op.getOperation().getContext(), attr);
      },
      "Returns the indexing_maps attribute for a linalg op.");
}
} // namespace linalg
} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir

NB_MODULE(_aiirDialectsLinalg, m) {
  m.doc() = "AIIR Linalg dialect.";

  aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN::linalg::
      populateDialectLinalgSubmodule(m);
}
