//===- DialectLLVM.cpp - Pybind module for LLVM dialect API support -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>
#include <vector>

#include "aiir-c/Dialect/LLVM.h"
#include "aiir-c/IR.h"
#include "aiir-c/Support.h"
#include "aiir-c/Target/LLVMIR.h"
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
namespace llvm {
//===--------------------------------------------------------------------===//
// StructType
//===--------------------------------------------------------------------===//

struct StructType : PyConcreteType<StructType> {
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsALLVMStructType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirLLVMStructTypeGetTypeID;
  static constexpr const char *pyClassName = "StructType";
  static inline const AiirStringRef name = aiirLLVMStructTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get_literal",
        [](const std::vector<PyType> &elements, bool packed,
           DefaultingPyLocation loc, DefaultingPyAiirContext context) {
          python::CollectDiagnosticsToStringScope scope(
              aiirLocationGetContext(loc));
          std::vector<AiirType> elements_(elements.size());
          std::copy(elements.begin(), elements.end(), elements_.begin());

          AiirType type = aiirLLVMStructTypeLiteralGetChecked(
              loc, elements.size(), elements_.data(), packed);
          if (aiirTypeIsNull(type)) {
            throw nb::value_error(scope.takeMessage().c_str());
          }
          return StructType(context->getRef(), type);
        },
        "elements"_a, nb::kw_only(), "packed"_a = false, "loc"_a = nb::none(),
        "context"_a = nb::none());

    c.def_static(
        "get_literal_unchecked",
        [](const std::vector<PyType> &elements, bool packed,
           DefaultingPyAiirContext context) {
          python::CollectDiagnosticsToStringScope scope(context.get()->get());

          std::vector<AiirType> elements_(elements.size());
          std::copy(elements.begin(), elements.end(), elements_.begin());

          AiirType type = aiirLLVMStructTypeLiteralGet(
              context.get()->get(), elements.size(), elements_.data(), packed);
          if (aiirTypeIsNull(type)) {
            throw nb::value_error(scope.takeMessage().c_str());
          }
          return StructType(context->getRef(), type);
        },
        "elements"_a, nb::kw_only(), "packed"_a = false,
        "context"_a = nb::none());

    c.def_static(
        "get_identified",
        [](const std::string &name, DefaultingPyAiirContext context) {
          return StructType(context->getRef(),
                            aiirLLVMStructTypeIdentifiedGet(
                                context.get()->get(),
                                aiirStringRefCreate(name.data(), name.size())));
        },
        "name"_a, nb::kw_only(), "context"_a = nb::none());

    c.def_static(
        "get_opaque",
        [](const std::string &name, DefaultingPyAiirContext context) {
          return StructType(context->getRef(),
                            aiirLLVMStructTypeOpaqueGet(
                                context.get()->get(),
                                aiirStringRefCreate(name.data(), name.size())));
        },
        "name"_a, "context"_a = nb::none());

    c.def(
        "set_body",
        [](const StructType &self, const std::vector<PyType> &elements,
           bool packed) {
          std::vector<AiirType> elements_(elements.size());
          std::copy(elements.begin(), elements.end(), elements_.begin());
          AiirLogicalResult result = aiirLLVMStructTypeSetBody(
              self, elements.size(), elements_.data(), packed);
          if (!aiirLogicalResultIsSuccess(result)) {
            throw nb::value_error(
                "Struct body already set to different content.");
          }
        },
        "elements"_a, nb::kw_only(), "packed"_a = false);

    c.def_static(
        "new_identified",
        [](const std::string &name, const std::vector<PyType> &elements,
           bool packed, DefaultingPyAiirContext context) {
          std::vector<AiirType> elements_(elements.size());
          std::copy(elements.begin(), elements.end(), elements_.begin());
          return StructType(context->getRef(),
                            aiirLLVMStructTypeIdentifiedNewGet(
                                context.get()->get(),
                                aiirStringRefCreate(name.data(), name.length()),
                                elements.size(), elements_.data(), packed));
        },
        "name"_a, "elements"_a, nb::kw_only(), "packed"_a = false,
        "context"_a = nb::none());

    c.def_prop_ro("name",
                  [](const StructType &type) -> std::optional<AiirStringRef> {
                    if (aiirLLVMStructTypeIsLiteral(type))
                      return std::nullopt;

                    return aiirLLVMStructTypeGetIdentifier(type);
                  });

    c.def_prop_ro("body", [](const StructType &type) -> nb::object {
      // Don't crash in absence of a body.
      if (aiirLLVMStructTypeIsOpaque(type))
        return nb::none();

      nb::list body;
      for (intptr_t i = 0, e = aiirLLVMStructTypeGetNumElementTypes(type);
           i < e; ++i) {
        body.append(aiirLLVMStructTypeGetElementType(type, i));
      }
      return body;
    });

    c.def_prop_ro("packed", [](const StructType &type) {
      return aiirLLVMStructTypeIsPacked(type);
    });

    c.def_prop_ro("opaque", [](const StructType &type) {
      return aiirLLVMStructTypeIsOpaque(type);
    });
  }
};

//===--------------------------------------------------------------------===//
// ArrayType
//===--------------------------------------------------------------------===//

struct ArrayType : PyConcreteType<ArrayType> {
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsALLVMArrayType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirLLVMArrayTypeGetTypeID;
  static constexpr const char *pyClassName = "ArrayType";
  static inline const AiirStringRef name = aiirLLVMArrayTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &elementType, unsigned numElements) {
          return ArrayType(elementType.getContext(),
                           aiirLLVMArrayTypeGet(elementType, numElements));
        },
        "element_type"_a, "num_elements"_a);
    c.def_prop_ro("element_type", [](const ArrayType &type) {
      return aiirLLVMArrayTypeGetElementType(type);
    });
    c.def_prop_ro("num_elements", [](const ArrayType &type) {
      return aiirLLVMArrayTypeGetNumElements(type);
    });
  }
};

//===--------------------------------------------------------------------===//
// PointerType
//===--------------------------------------------------------------------===//

struct PointerType : PyConcreteType<PointerType> {
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsALLVMPointerType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirLLVMPointerTypeGetTypeID;
  static constexpr const char *pyClassName = "PointerType";
  static inline const AiirStringRef name = aiirLLVMPointerTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::optional<unsigned> addressSpace,
           DefaultingPyAiirContext context) {
          python::CollectDiagnosticsToStringScope scope(context.get()->get());
          AiirType type = aiirLLVMPointerTypeGet(
              context.get()->get(),
              addressSpace.has_value() ? *addressSpace : 0);
          if (aiirTypeIsNull(type)) {
            throw nb::value_error(scope.takeMessage().c_str());
          }
          return PointerType(context->getRef(), type);
        },
        "address_space"_a = nb::none(), nb::kw_only(),
        "context"_a = nb::none());
    c.def_prop_ro("address_space", [](const PointerType &type) {
      return aiirLLVMPointerTypeGetAddressSpace(type);
    });
  }
};

//===--------------------------------------------------------------------===//
// FunctionType
//===--------------------------------------------------------------------===//

struct FunctionType : PyConcreteType<FunctionType> {
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsALLVMFunctionType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirLLVMFunctionTypeGetTypeID;
  static constexpr const char *pyClassName = "FunctionType";
  static inline const AiirStringRef name = aiirLLVMFunctionTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &resultType, const std::vector<PyType> &argumentTypes,
           bool isVarArg) {
          std::vector<AiirType> argTypes(argumentTypes.size());
          std::copy(argumentTypes.begin(), argumentTypes.end(),
                    argTypes.begin());
          return FunctionType(
              resultType.getContext(),
              aiirLLVMFunctionTypeGet(resultType, argTypes.size(),
                                      argTypes.data(), isVarArg));
        },
        "result_type"_a, "argument_types"_a, nb::kw_only(),
        "is_var_arg"_a = false);
    c.def_prop_ro("return_type", [](const FunctionType &type) {
      return aiirLLVMFunctionTypeGetReturnType(type);
    });
    c.def_prop_ro("num_inputs", [](const FunctionType &type) {
      return aiirLLVMFunctionTypeGetNumInputs(type);
    });
    c.def_prop_ro("inputs", [](const FunctionType &type) {
      nb::list inputs;
      for (intptr_t i = 0, e = aiirLLVMFunctionTypeGetNumInputs(type); i < e;
           ++i) {
        inputs.append(aiirLLVMFunctionTypeGetInput(type, i));
      }
      return inputs;
    });
    c.def_prop_ro("is_var_arg", [](const FunctionType &type) {
      return aiirLLVMFunctionTypeIsVarArg(type);
    });
  }
};

//===--------------------------------------------------------------------===//
// Metadata Attributes
//===--------------------------------------------------------------------===//

struct MDStringAttr : PyConcreteAttribute<MDStringAttr> {
  static constexpr IsAFunctionTy isaFunction = aiirLLVMAttrIsAMDStringAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirLLVMMDStringAttrGetTypeID;
  static constexpr const char *pyClassName = "MDStringAttr";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::string &value, DefaultingPyAiirContext context) {
          return MDStringAttr(
              context->getRef(),
              aiirLLVMMDStringAttrGet(
                  context.get()->get(),
                  aiirStringRefCreate(value.data(), value.size())));
        },
        "value"_a, nb::kw_only(), "context"_a = nb::none());
    c.def_prop_ro("value", [](const MDStringAttr &self) {
      AiirStringRef ref = aiirLLVMMDStringAttrGetValue(self);
      return nb::str(ref.data, ref.length);
    });
  }
};

struct MDConstantAttr : PyConcreteAttribute<MDConstantAttr> {
  static constexpr IsAFunctionTy isaFunction = aiirLLVMAttrIsAMDConstantAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirLLVMMDConstantAttrGetTypeID;
  static constexpr const char *pyClassName = "MDConstantAttr";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyAttribute &valueAttr, DefaultingPyAiirContext context) {
          return MDConstantAttr(
              context->getRef(),
              aiirLLVMMDConstantAttrGet(context.get()->get(), valueAttr));
        },
        "value"_a, nb::kw_only(), "context"_a = nb::none());
    c.def_prop_ro("value", [](const MDConstantAttr &self) {
      return aiirLLVMMDConstantAttrGetValue(self);
    });
  }
};

struct MDFuncAttr : PyConcreteAttribute<MDFuncAttr> {
  static constexpr IsAFunctionTy isaFunction = aiirLLVMAttrIsAMDFuncAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirLLVMMDFuncAttrGetTypeID;
  static constexpr const char *pyClassName = "MDFuncAttr";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::string &name, DefaultingPyAiirContext context) {
          AiirAttribute symRef = aiirFlatSymbolRefAttrGet(
              context.get()->get(),
              aiirStringRefCreate(name.data(), name.size()));
          return MDFuncAttr(
              context->getRef(),
              aiirLLVMMDFuncAttrGet(context.get()->get(), symRef));
        },
        "name"_a, nb::kw_only(), "context"_a = nb::none());
    c.def_prop_ro("name", [](const MDFuncAttr &self) {
      AiirAttribute symRef = aiirLLVMMDFuncAttrGetName(self);
      AiirStringRef ref = aiirFlatSymbolRefAttrGetValue(symRef);
      return nb::str(ref.data, ref.length);
    });
  }
};

struct MDNodeAttr : PyConcreteAttribute<MDNodeAttr> {
  static constexpr IsAFunctionTy isaFunction = aiirLLVMAttrIsAMDNodeAttr;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirLLVMMDNodeAttrGetTypeID;
  static constexpr const char *pyClassName = "MDNodeAttr";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::vector<PyAttribute> &operands,
           DefaultingPyAiirContext context) {
          std::vector<AiirAttribute> operands_(operands.size());
          std::copy(operands.begin(), operands.end(), operands_.begin());
          return MDNodeAttr(context->getRef(),
                            aiirLLVMMDNodeAttrGet(context.get()->get(),
                                                  operands_.size(),
                                                  operands_.data()));
        },
        "operands"_a, nb::kw_only(), "context"_a = nb::none());
    c.def_prop_ro("num_operands", [](const MDNodeAttr &self) {
      return aiirLLVMMDNodeAttrGetNumOperands(self);
    });
    c.def("__getitem__", [](const MDNodeAttr &self, intptr_t index) {
      intptr_t n = aiirLLVMMDNodeAttrGetNumOperands(self);
      if (index < 0 || index >= n)
        throw nb::index_error("MDNodeAttr operand index out of range");
      return aiirLLVMMDNodeAttrGetOperand(self, index);
    });
    c.def("__len__", [](const MDNodeAttr &self) {
      return aiirLLVMMDNodeAttrGetNumOperands(self);
    });
  }
};

static void populateDialectLLVMSubmodule(nanobind::module_ &m) {
  StructType::bind(m);
  ArrayType::bind(m);
  PointerType::bind(m);
  FunctionType::bind(m);
  MDStringAttr::bind(m);
  MDConstantAttr::bind(m);
  MDFuncAttr::bind(m);
  MDNodeAttr::bind(m);

  m.def(
      "translate_module_to_llvmir",
      [](const PyOperation &module) {
        return aiirTranslateModuleToLLVMIRToString(module);
      },
      "module"_a, nb::rv_policy::take_ownership);
}
} // namespace llvm
} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir

NB_MODULE(_aiirDialectsLLVM, m) {
  m.doc() = "AIIR LLVM Dialect";

  aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN::llvm::populateDialectLLVMSubmodule(
      m);
}
