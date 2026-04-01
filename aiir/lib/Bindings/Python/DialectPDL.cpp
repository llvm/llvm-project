//===- DialectPDL.cpp - 'pdl' dialect submodule ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/PDL.h"
#include "aiir-c/IR.h"
#include "aiir/Bindings/Python/IRCore.h"
#include "aiir/Bindings/Python/Nanobind.h"
#include "aiir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;
using namespace aiir::python::nanobind_adaptors;

namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {
namespace pdl {

//===-------------------------------------------------------------------===//
// PDLType
//===-------------------------------------------------------------------===//

struct PDLType : PyConcreteType<PDLType> {
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAPDLType;
  static constexpr const char *pyClassName = "PDLType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {}
};

//===-------------------------------------------------------------------===//
// AttributeType
//===-------------------------------------------------------------------===//

struct AttributeType : PyConcreteType<AttributeType> {
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAPDLAttributeType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirPDLAttributeTypeGetTypeID;
  static constexpr const char *pyClassName = "AttributeType";
  static inline const AiirStringRef name = aiirPDLAttributeTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyAiirContext context) {
          return AttributeType(context->getRef(),
                               aiirPDLAttributeTypeGet(context.get()->get()));
        },
        "Get an instance of AttributeType in given context.",
        nb::arg("context").none() = nb::none());
  }
};

//===-------------------------------------------------------------------===//
// OperationType
//===-------------------------------------------------------------------===//

struct OperationType : PyConcreteType<OperationType> {
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAPDLOperationType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirPDLOperationTypeGetTypeID;
  static constexpr const char *pyClassName = "OperationType";
  static inline const AiirStringRef name = aiirPDLOperationTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyAiirContext context) {
          return OperationType(context->getRef(),
                               aiirPDLOperationTypeGet(context.get()->get()));
        },
        "Get an instance of OperationType in given context.",
        nb::arg("context").none() = nb::none());
  }
};

//===-------------------------------------------------------------------===//
// RangeType
//===-------------------------------------------------------------------===//

struct RangeType : PyConcreteType<RangeType> {
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAPDLRangeType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirPDLRangeTypeGetTypeID;
  static constexpr const char *pyClassName = "RangeType";
  static inline const AiirStringRef name = aiirPDLRangeTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const PyType &elementType, DefaultingPyAiirContext context) {
          return RangeType(context->getRef(), aiirPDLRangeTypeGet(elementType));
        },
        "Gets an instance of RangeType in the same context as the provided "
        "element type.",
        nb::arg("element_type"), nb::arg("context").none() = nb::none());
    c.def_prop_ro(
        "element_type",
        [](RangeType &type) {
          return PyType(type.getContext(), aiirPDLRangeTypeGetElementType(type))
              .maybeDownCast();
        },
        "Get the element type.");
  }
};

//===-------------------------------------------------------------------===//
// TypeType
//===-------------------------------------------------------------------===//

struct TypeType : PyConcreteType<TypeType> {
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAPDLTypeType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirPDLTypeTypeGetTypeID;
  static constexpr const char *pyClassName = "TypeType";
  static inline const AiirStringRef name = aiirPDLTypeTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyAiirContext context) {
          return TypeType(context->getRef(),
                          aiirPDLTypeTypeGet(context.get()->get()));
        },
        "Get an instance of TypeType in given context.",
        nb::arg("context").none() = nb::none());
  }
};

//===-------------------------------------------------------------------===//
// ValueType
//===-------------------------------------------------------------------===//

struct ValueType : PyConcreteType<ValueType> {
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsAPDLValueType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirPDLValueTypeGetTypeID;
  static constexpr const char *pyClassName = "ValueType";
  static inline const AiirStringRef name = aiirPDLValueTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyAiirContext context) {
          return ValueType(context->getRef(),
                           aiirPDLValueTypeGet(context.get()->get()));
        },
        "Get an instance of TypeType in given context.",
        nb::arg("context").none() = nb::none());
  }
};

static void populateDialectPDLSubmodule(nanobind::module_ &m) {
  PDLType::bind(m);
  AttributeType::bind(m);
  OperationType::bind(m);
  RangeType::bind(m);
  TypeType::bind(m);
  ValueType::bind(m);
}
} // namespace pdl
} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir

NB_MODULE(_aiirDialectsPDL, m) {
  m.doc() = "AIIR PDL dialect.";
  aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN::pdl::populateDialectPDLSubmodule(
      m);
}
