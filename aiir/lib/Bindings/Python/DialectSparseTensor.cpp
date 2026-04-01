//===- DialectSparseTensor.cpp - 'sparse_tensor' dialect submodule --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <optional>
#include <vector>

#include "aiir-c/AffineMap.h"
#include "aiir-c/Dialect/SparseTensor.h"
#include "aiir-c/IR.h"
#include "aiir/Bindings/Python/IRCore.h"
#include "aiir/Bindings/Python/Nanobind.h"
#include "aiir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;
using namespace aiir::python::nanobind_adaptors;

namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {
namespace sparse_tensor {

enum class PySparseTensorLevelFormat : std::underlying_type_t<
    AiirSparseTensorLevelFormat> {
  DENSE = AIIR_SPARSE_TENSOR_LEVEL_DENSE,
  N_OUT_OF_M = AIIR_SPARSE_TENSOR_LEVEL_N_OUT_OF_M,
  COMPRESSED = AIIR_SPARSE_TENSOR_LEVEL_COMPRESSED,
  SINGLETON = AIIR_SPARSE_TENSOR_LEVEL_SINGLETON,
  LOOSE_COMPRESSED = AIIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED
};

enum class PySparseTensorLevelPropertyNondefault : std::underlying_type_t<
    AiirSparseTensorLevelPropertyNondefault> {
  NON_ORDERED = AIIR_SPARSE_PROPERTY_NON_ORDERED,
  NON_UNIQUE = AIIR_SPARSE_PROPERTY_NON_UNIQUE,
  SOA = AIIR_SPARSE_PROPERTY_SOA,
};

struct EncodingAttr : PyConcreteAttribute<EncodingAttr> {
  static constexpr IsAFunctionTy isaFunction =
      aiirAttributeIsASparseTensorEncodingAttr;
  static constexpr const char *pyClassName = "EncodingAttr";
  static inline const AiirStringRef name =
      aiirSparseTensorEncodingAttrGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::vector<AiirSparseTensorLevelType> lvlTypes,
           std::optional<PyAffineMap> dimToLvl,
           std::optional<PyAffineMap> lvlToDim, int posWidth, int crdWidth,
           std::optional<PyAttribute> explicitVal,
           std::optional<PyAttribute> implicitVal,
           DefaultingPyAiirContext context) {
          return EncodingAttr(
              context->getRef(),
              aiirSparseTensorEncodingAttrGet(
                  context.get()->get(), lvlTypes.size(), lvlTypes.data(),
                  dimToLvl ? *dimToLvl : AiirAffineMap{nullptr},
                  lvlToDim ? *lvlToDim : AiirAffineMap{nullptr}, posWidth,
                  crdWidth, explicitVal ? *explicitVal : AiirAttribute{nullptr},
                  implicitVal ? *implicitVal : AiirAttribute{nullptr}));
        },
        nb::arg("lvl_types"), nb::arg("dim_to_lvl").none(),
        nb::arg("lvl_to_dim").none(), nb::arg("pos_width"),
        nb::arg("crd_width"), nb::arg("explicit_val") = nb::none(),
        nb::arg("implicit_val") = nb::none(), nb::arg("context") = nb::none(),
        "Gets a sparse_tensor.encoding from parameters.");

    c.def_static(
        "build_level_type",
        [](PySparseTensorLevelFormat lvlFmt,
           const std::vector<PySparseTensorLevelPropertyNondefault> &properties,
           unsigned n, unsigned m) {
          std::vector<AiirSparseTensorLevelPropertyNondefault> props;
          props.reserve(properties.size());
          for (auto prop : properties) {
            props.push_back(
                static_cast<AiirSparseTensorLevelPropertyNondefault>(prop));
          }
          return aiirSparseTensorEncodingAttrBuildLvlType(
              static_cast<AiirSparseTensorLevelFormat>(lvlFmt), props.data(),
              props.size(), n, m);
        },
        nb::arg("lvl_fmt"),
        nb::arg("properties") =
            std::vector<PySparseTensorLevelPropertyNondefault>(),
        nb::arg("n") = 0, nb::arg("m") = 0,
        "Builds a sparse_tensor.encoding.level_type from parameters.");

    c.def_prop_ro("lvl_types", [](const EncodingAttr &self) {
      const int lvlRank = aiirSparseTensorEncodingGetLvlRank(self);
      std::vector<AiirSparseTensorLevelType> ret;
      ret.reserve(lvlRank);
      for (int l = 0; l < lvlRank; ++l)
        ret.push_back(aiirSparseTensorEncodingAttrGetLvlType(self, l));
      return ret;
    });

    c.def_prop_ro(
        "dim_to_lvl", [](EncodingAttr &self) -> std::optional<PyAffineMap> {
          AiirAffineMap ret = aiirSparseTensorEncodingAttrGetDimToLvl(self);
          if (aiirAffineMapIsNull(ret))
            return {};
          return PyAffineMap(self.getContext(), ret);
        });

    c.def_prop_ro(
        "lvl_to_dim", [](EncodingAttr &self) -> std::optional<PyAffineMap> {
          AiirAffineMap ret = aiirSparseTensorEncodingAttrGetLvlToDim(self);
          if (aiirAffineMapIsNull(ret))
            return {};
          return PyAffineMap(self.getContext(), ret);
        });

    c.def_prop_ro("pos_width", aiirSparseTensorEncodingAttrGetPosWidth);
    c.def_prop_ro("crd_width", aiirSparseTensorEncodingAttrGetCrdWidth);

    c.def_prop_ro("explicit_val",
                  [](EncodingAttr &self)
                      -> std::optional<nb::typed<nb::object, PyAttribute>> {
                    AiirAttribute ret =
                        aiirSparseTensorEncodingAttrGetExplicitVal(self);
                    if (aiirAttributeIsNull(ret))
                      return {};
                    return PyAttribute(self.getContext(), ret).maybeDownCast();
                  });

    c.def_prop_ro("implicit_val",
                  [](EncodingAttr &self)
                      -> std::optional<nb::typed<nb::object, PyAttribute>> {
                    AiirAttribute ret =
                        aiirSparseTensorEncodingAttrGetImplicitVal(self);
                    if (aiirAttributeIsNull(ret))
                      return {};
                    return PyAttribute(self.getContext(), ret).maybeDownCast();
                  });

    c.def_prop_ro("structured_n", [](const EncodingAttr &self) -> unsigned {
      const int lvlRank = aiirSparseTensorEncodingGetLvlRank(self);
      return aiirSparseTensorEncodingAttrGetStructuredN(
          aiirSparseTensorEncodingAttrGetLvlType(self, lvlRank - 1));
    });

    c.def_prop_ro("structured_m", [](const EncodingAttr &self) -> unsigned {
      const int lvlRank = aiirSparseTensorEncodingGetLvlRank(self);
      return aiirSparseTensorEncodingAttrGetStructuredM(
          aiirSparseTensorEncodingAttrGetLvlType(self, lvlRank - 1));
    });

    c.def_prop_ro("lvl_formats_enum", [](const EncodingAttr &self) {
      const int lvlRank = aiirSparseTensorEncodingGetLvlRank(self);
      std::vector<PySparseTensorLevelFormat> ret;
      ret.reserve(lvlRank);

      for (int l = 0; l < lvlRank; l++)
        ret.push_back(static_cast<PySparseTensorLevelFormat>(
            aiirSparseTensorEncodingAttrGetLvlFmt(self, l)));
      return ret;
    });
  }
};

static void populateDialectSparseTensorSubmodule(nb::module_ &m) {
  nb::enum_<PySparseTensorLevelFormat>(m, "LevelFormat", nb::is_arithmetic(),
                                       nb::is_flag())
      .value("dense", PySparseTensorLevelFormat::DENSE)
      .value("n_out_of_m", PySparseTensorLevelFormat::N_OUT_OF_M)
      .value("compressed", PySparseTensorLevelFormat::COMPRESSED)
      .value("singleton", PySparseTensorLevelFormat::SINGLETON)
      .value("loose_compressed", PySparseTensorLevelFormat::LOOSE_COMPRESSED);
  nb::enum_<PySparseTensorLevelPropertyNondefault>(m, "LevelProperty")
      .value("non_ordered", PySparseTensorLevelPropertyNondefault::NON_ORDERED)
      .value("non_unique", PySparseTensorLevelPropertyNondefault::NON_UNIQUE)
      .value("soa", PySparseTensorLevelPropertyNondefault::SOA);

  EncodingAttr::bind(m);
}
} // namespace sparse_tensor
} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir

NB_MODULE(_aiirDialectsSparseTensor, m) {
  m.doc() = "AIIR SparseTensor dialect.";
  aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN::sparse_tensor::
      populateDialectSparseTensorSubmodule(m);
}
