//===- DialectTransform.cpp - 'transform' dialect submodule ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>

#include "Rewrite.h"
#include "aiir-c/Dialect/Transform.h"
#include "aiir-c/IR.h"
#include "aiir-c/Rewrite.h"
#include "aiir-c/Support.h"
#include "aiir/Bindings/Python/IRCore.h"
#include "aiir/Bindings/Python/IRInterfaces.h"
#include "nanobind/nanobind.h"
#include <nanobind/trampoline.h>

namespace nb = nanobind;
using namespace aiir::python::nanobind_adaptors;

namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {
namespace transform {

//===----------------------------------------------------------------------===//
// TransformRewriter
//===----------------------------------------------------------------------===//
class PyTransformRewriter : public PyRewriterBase<PyTransformRewriter> {
public:
  static constexpr const char *pyClassName = "TransformRewriter";

  PyTransformRewriter(AiirTransformRewriter rewriter)
      : PyRewriterBase(aiirTransformRewriterAsBase(rewriter)) {}
};

//===----------------------------------------------------------------------===//
// TransformResults
//===----------------------------------------------------------------------===//
class PyTransformResults {
public:
  PyTransformResults(AiirTransformResults results) : results(results) {}

  AiirTransformResults get() const { return results; }

  void setOps(PyValue &result,
              const nb::typed<nb::sequence, PyOperationBase> &ops) {
    std::vector<AiirOperation> opsVec;
    opsVec.reserve(nb::len(ops));
    for (auto op : ops) {
      opsVec.push_back(nb::cast<AiirOperation>(op));
    }
    aiirTransformResultsSetOps(results, result, opsVec.size(), opsVec.data());
  }

  void setValues(PyValue &result,
                 const nb::typed<nb::sequence, PyValue> &values) {
    std::vector<AiirValue> valuesVec;
    valuesVec.reserve(nb::len(values));
    for (auto item : values) {
      valuesVec.push_back(nb::cast<AiirValue>(item));
    }
    aiirTransformResultsSetValues(results, result, valuesVec.size(),
                                  valuesVec.data());
  }

  void setParams(PyValue &result,
                 const nb::typed<nb::sequence, PyAttribute> &params) {
    std::vector<AiirAttribute> paramsVec;
    paramsVec.reserve(nb::len(params));
    for (auto item : params) {
      paramsVec.push_back(nb::cast<AiirAttribute>(item));
    }
    aiirTransformResultsSetParams(results, result, paramsVec.size(),
                                  paramsVec.data());
  }

  static void bind(nanobind::module_ &m) {
    nb::class_<PyTransformResults>(m, "TransformResults")
        .def(nb::init<AiirTransformResults>())
        .def("set_ops", &PyTransformResults::setOps,
             "Set the payload operations for a transform result.",
             nb::arg("result"), nb::arg("ops"))
        .def("set_values", &PyTransformResults::setValues,
             "Set the payload values for a transform result.",
             nb::arg("result"), nb::arg("values"))
        .def("set_params", &PyTransformResults::setParams,
             "Set the parameters for a transform result.", nb::arg("result"),
             nb::arg("params"));
  }

private:
  AiirTransformResults results;
};

//===----------------------------------------------------------------------===//
// TransformState
//===----------------------------------------------------------------------===//
class PyTransformState {
public:
  PyTransformState(AiirTransformState state) : state(state) {}

  AiirTransformState get() const { return state; }

  static void bind(nanobind::module_ &m) {
    nb::class_<PyTransformState>(m, "TransformState")
        .def(nb::init<AiirTransformState>())
        .def("get_payload_ops", &PyTransformState::getPayloadOps,
             "Get the payload operations associated with a transform IR value.",
             nb::arg("operand"))
        .def("get_payload_values", &PyTransformState::getPayloadValues,
             "Get the payload values associated with a transform IR value.",
             nb::arg("operand"))
        .def("get_params", &PyTransformState::getParams,
             "Get the parameters (attributes) associated with a transform IR "
             "value.",
             nb::arg("operand"));
  }

private:
  nanobind::list getPayloadOps(PyValue &value) {
    nanobind::list result;
    aiirTransformStateForEachPayloadOp(
        state, value,
        [](AiirOperation op, void *userData) {
          PyAiirContextRef context =
              PyAiirContext::forContext(aiirOperationGetContext(op));
          auto opview = PyOperation::forOperation(context, op)->createOpView();
          static_cast<nanobind::list *>(userData)->append(opview);
        },
        &result);
    return result;
  }

  nanobind::list getPayloadValues(PyValue &value) {
    nanobind::list result;
    aiirTransformStateForEachPayloadValue(
        state, value,
        [](AiirValue val, void *userData) {
          static_cast<nanobind::list *>(userData)->append(val);
        },
        &result);
    return result;
  }

  nanobind::list getParams(PyValue &value) {
    nanobind::list result;
    aiirTransformStateForEachParam(
        state, value,
        [](AiirAttribute attr, void *userData) {
          static_cast<nanobind::list *>(userData)->append(attr);
        },
        &result);
    return result;
  }

  AiirTransformState state;
};

//===----------------------------------------------------------------------===//
// TransformOpInterface
//===----------------------------------------------------------------------===//
class PyTransformOpInterface
    : public PyConcreteOpInterface<PyTransformOpInterface> {
public:
  using PyConcreteOpInterface<PyTransformOpInterface>::PyConcreteOpInterface;

  constexpr static const char *pyClassName = "TransformOpInterface";
  constexpr static GetTypeIDFunctionTy getInterfaceID =
      &aiirTransformOpInterfaceTypeID;

  /// Attach a new TransformOpInterface FallbackModel to the named operation.
  /// The FallbackModel acts as a trampoline for callbacks on the Python class.
  static void attach(nb::object &target, const std::string &opName,
                     DefaultingPyAiirContext ctx) {
    // Prepare the callbacks that will be used by the FallbackModel.
    AiirTransformOpInterfaceCallbacks callbacks;
    // Make the pointer to the Python class available to the callbacks.
    callbacks.userData = target.ptr();
    nb::handle(static_cast<PyObject *>(callbacks.userData)).inc_ref();

    // The above ref bump is all we need as initialization, no need to run the
    // construct callback.
    callbacks.construct = nullptr;
    // Upon the FallbackModel's destruction, drop the ref to the Python class.
    callbacks.destruct = [](void *userData) {
      nb::handle(static_cast<PyObject *>(userData)).dec_ref();
    };
    // The apply callback which calls into Python.
    callbacks.apply = [](AiirOperation op, AiirTransformRewriter rewriter,
                         AiirTransformResults results, AiirTransformState state,
                         void *userData) -> AiirDiagnosedSilenceableFailure {
      nb::handle pyClass(static_cast<PyObject *>(userData));

      auto pyApply = nb::cast<nb::callable>(nb::getattr(pyClass, "apply"));

      auto pyRewriter = PyTransformRewriter(rewriter);
      auto pyResults = PyTransformResults(results);
      auto pyState = PyTransformState(state);

      // Invoke `pyClass.apply(opview(op), rewriter, results, state)` as a
      // staticmethod.
      PyAiirContextRef context =
          PyAiirContext::forContext(aiirOperationGetContext(op));
      auto opview = PyOperation::forOperation(context, op)->createOpView();
      nb::object res = pyApply(opview, pyRewriter, pyResults, pyState);

      return nb::cast<AiirDiagnosedSilenceableFailure>(res);
    };

    // The allows_repeated_handle_operands callback which calls into Python.
    callbacks.allowsRepeatedHandleOperands = [](AiirOperation op,
                                                void *userData) -> bool {
      nb::handle pyClass(static_cast<PyObject *>(userData));

      auto pyAllowRepeatedHandleOperands = nb::cast<nb::callable>(
          nb::getattr(pyClass, "allow_repeated_handle_operands"));

      // Invoke `pyClass.allow_repeated_handle_operands(opview(op))` as a
      // staticmethod.
      PyAiirContextRef context =
          PyAiirContext::forContext(aiirOperationGetContext(op));
      auto opview = PyOperation::forOperation(context, op)->createOpView();
      nb::object res = pyAllowRepeatedHandleOperands(opview);

      return nb::cast<bool>(res);
    };

    // Attach a FallbackModel, which calls into Python, to the named operation.
    aiirTransformOpInterfaceAttachFallbackModel(
        ctx->get(), aiirStringRefCreate(opName.c_str(), opName.size()),
        callbacks);
  }

  static void bindDerived(ClassTy &cls) {
    cls.attr("attach") = classmethod(
        [](const nb::object &cls, const nb::object &opName, nb::object target,
           DefaultingPyAiirContext context) {
          if (target.is_none())
            target = cls;
          return attach(target, nb::cast<std::string>(opName), context);
        },
        nb::arg("cls"), nb::arg("op_name"), nb::kw_only(),
        nb::arg("target").none() = nb::none(),
        nb::arg("context").none() = nb::none(),
        "Attach the interface subclass to the given operation name.");
  }
};

//===----------------------------------------------------------------------===//
// PatternDescriptorOpInterface
//===----------------------------------------------------------------------===//
class PyPatternDescriptorOpInterface
    : public PyConcreteOpInterface<PyPatternDescriptorOpInterface> {
public:
  using PyConcreteOpInterface<
      PyPatternDescriptorOpInterface>::PyConcreteOpInterface;

  constexpr static const char *pyClassName = "PatternDescriptorOpInterface";
  constexpr static GetTypeIDFunctionTy getInterfaceID =
      &aiirPatternDescriptorOpInterfaceTypeID;

  /// Attach a new PatternDescriptorOpInterface FallbackModel to the named
  /// operation. The FallbackModel acts as a trampoline for callbacks on the
  /// Python class.
  static void attach(nb::object &target, const std::string &opName,
                     DefaultingPyAiirContext ctx) {
    // Prepare the callbacks that will be used by the FallbackModel.
    AiirPatternDescriptorOpInterfaceCallbacks callbacks;
    // Make the pointer to the Python class available to the callbacks.
    callbacks.userData = target.ptr();
    nb::handle(static_cast<PyObject *>(callbacks.userData)).inc_ref();

    // The above ref bump is all we need as initialization, no need to run the
    // construct callback.
    callbacks.construct = nullptr;
    // Upon the FallbackModel's destruction, drop the ref to the Python class.
    callbacks.destruct = [](void *userData) {
      nb::handle(static_cast<PyObject *>(userData)).dec_ref();
    };

    // The populatePatterns callback which calls into Python.
    callbacks.populatePatterns =
        [](AiirOperation op, AiirRewritePatternSet patterns, void *userData) {
          nb::handle pyClass(static_cast<PyObject *>(userData));

          auto pyPopulatePatterns =
              nb::cast<nb::callable>(nb::getattr(pyClass, "populate_patterns"));

          auto pyPatterns = PyRewritePatternSet(patterns);

          // Invoke `pyClass.populate_patterns(opview(op), patterns)` as a
          // staticmethod.
          AiirContext ctx = aiirOperationGetContext(op);
          PyAiirContextRef context = PyAiirContext::forContext(ctx);
          auto opview = PyOperation::forOperation(context, op)->createOpView();
          pyPopulatePatterns(opview, pyPatterns);
        };

    // The populatePatternsWithState callback which calls into Python.
    // Check if the Python class has populate_patterns_with_state method.
    if (nb::hasattr(target, "populate_patterns_with_state")) {
      callbacks.populatePatternsWithState = [](AiirOperation op,
                                               AiirRewritePatternSet patterns,
                                               AiirTransformState state,
                                               void *userData) {
        nb::handle pyClass(static_cast<PyObject *>(userData));

        auto pyPopulatePatternsWithState = nb::cast<nb::callable>(
            nb::getattr(pyClass, "populate_patterns_with_state"));

        auto pyPatterns = PyRewritePatternSet(patterns);
        auto pyState = PyTransformState(state);

        // Invoke `pyClass.populate_patterns_with_state(opview(op), patterns,
        // state)` as a staticmethod.
        AiirContext ctx = aiirOperationGetContext(op);
        PyAiirContextRef context = PyAiirContext::forContext(ctx);
        auto opview = PyOperation::forOperation(context, op)->createOpView();
        pyPopulatePatternsWithState(opview, pyPatterns, pyState);
      };
    } else {
      // Use default implementation (will call populatePatterns).
      callbacks.populatePatternsWithState = nullptr;
    }

    // Attach a FallbackModel, which calls into Python, to the named operation.
    aiirPatternDescriptorOpInterfaceAttachFallbackModel(
        ctx->get(), aiirStringRefCreate(opName.c_str(), opName.size()),
        callbacks);
  }

  static void bindDerived(ClassTy &cls) {
    cls.attr("attach") = classmethod(
        [](const nb::object &cls, const nb::object &opName, nb::object target,
           DefaultingPyAiirContext context) {
          if (target.is_none())
            target = cls;
          return attach(target, nb::cast<std::string>(opName), context);
        },
        nb::arg("cls"), nb::arg("op_name"), nb::kw_only(),
        nb::arg("target").none() = nb::none(),
        nb::arg("context").none() = nb::none(),
        "Attach the interface subclass to the given operation name.");
  }
};

//===-------------------------------------------------------------------===//
// AnyOpType
//===-------------------------------------------------------------------===//

struct AnyOpType : PyConcreteType<AnyOpType> {
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsATransformAnyOpType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirTransformAnyOpTypeGetTypeID;
  static constexpr const char *pyClassName = "AnyOpType";
  static inline const AiirStringRef name = aiirTransformAnyOpTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyAiirContext context) {
          return AnyOpType(context->getRef(),
                           aiirTransformAnyOpTypeGet(context.get()->get()));
        },
        "Get an instance of AnyOpType in the given context.",
        nb::arg("context").none() = nb::none());
  }
};

//===-------------------------------------------------------------------===//
// AnyParamType
//===-------------------------------------------------------------------===//

struct AnyParamType : PyConcreteType<AnyParamType> {
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsATransformAnyParamType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirTransformAnyParamTypeGetTypeID;
  static constexpr const char *pyClassName = "AnyParamType";
  static inline const AiirStringRef name = aiirTransformAnyParamTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyAiirContext context) {
          return AnyParamType(context->getRef(), aiirTransformAnyParamTypeGet(
                                                     context.get()->get()));
        },
        "Get an instance of AnyParamType in the given context.",
        nb::arg("context").none() = nb::none());
  }
};

//===-------------------------------------------------------------------===//
// AnyValueType
//===-------------------------------------------------------------------===//

struct AnyValueType : PyConcreteType<AnyValueType> {
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsATransformAnyValueType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirTransformAnyValueTypeGetTypeID;
  static constexpr const char *pyClassName = "AnyValueType";
  static inline const AiirStringRef name = aiirTransformAnyValueTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyAiirContext context) {
          return AnyValueType(context->getRef(), aiirTransformAnyValueTypeGet(
                                                     context.get()->get()));
        },
        "Get an instance of AnyValueType in the given context.",
        nb::arg("context").none() = nb::none());
  }
};

//===-------------------------------------------------------------------===//
// OperationType
//===-------------------------------------------------------------------===//

struct OperationType : PyConcreteType<OperationType> {
  static constexpr IsAFunctionTy isaFunction =
      aiirTypeIsATransformOperationType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirTransformOperationTypeGetTypeID;
  static constexpr const char *pyClassName = "OperationType";
  static inline const AiirStringRef name = aiirTransformOperationTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::string &operationName, DefaultingPyAiirContext context) {
          AiirStringRef cOperationName =
              aiirStringRefCreate(operationName.data(), operationName.size());
          return OperationType(context->getRef(),
                               aiirTransformOperationTypeGet(
                                   context.get()->get(), cOperationName));
        },
        "Get an instance of OperationType for the given kind in the given "
        "context",
        nb::arg("operation_name"), nb::arg("context").none() = nb::none());
    c.def_prop_ro(
        "operation_name",
        [](const OperationType &type) {
          AiirStringRef operationName =
              aiirTransformOperationTypeGetOperationName(type);
          return nb::str(operationName.data, operationName.length);
        },
        "Get the name of the payload operation accepted by the handle.");
  }
};

//===-------------------------------------------------------------------===//
// ParamType
//===-------------------------------------------------------------------===//

struct ParamType : PyConcreteType<ParamType> {
  static constexpr IsAFunctionTy isaFunction = aiirTypeIsATransformParamType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      aiirTransformParamTypeGetTypeID;
  static constexpr const char *pyClassName = "ParamType";
  static inline const AiirStringRef name = aiirTransformParamTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const PyType &type, DefaultingPyAiirContext context) {
          return ParamType(context->getRef(), aiirTransformParamTypeGet(
                                                  context.get()->get(), type));
        },
        "Get an instance of ParamType for the given type in the given context.",
        nb::arg("type"), nb::arg("context").none() = nb::none());
    c.def_prop_ro(
        "type",
        [](ParamType type) {
          return PyType(type.getContext(), aiirTransformParamTypeGetType(type))
              .maybeDownCast();
        },
        "Get the type this ParamType is associated with.");
  }
};

//===----------------------------------------------------------------------===//
// MemoryEffectsOpInterface helpers
//===----------------------------------------------------------------------===//

namespace {
void onlyReadsHandle(nb::iterable &operands,
                     PyMemoryEffectsInstanceList effects) {
  std::vector<AiirOpOperand> operandsVec;
  for (auto operand : operands)
    operandsVec.push_back(nb::cast<PyOpOperand>(operand));
  aiirTransformOnlyReadsHandle(operandsVec.data(), operandsVec.size(),
                               effects.effects);
};

void consumesHandle(nb::iterable &operands,
                    PyMemoryEffectsInstanceList effects) {
  std::vector<AiirOpOperand> operandsVec;
  for (auto operand : operands)
    operandsVec.push_back(nb::cast<PyOpOperand>(operand));
  aiirTransformConsumesHandle(operandsVec.data(), operandsVec.size(),
                              effects.effects);
};

void producesHandle(nb::iterable &results,
                    PyMemoryEffectsInstanceList effects) {
  std::vector<AiirValue> resultsVec;
  for (auto result : results)
    resultsVec.push_back(nb::cast<PyOpResult>(result).get());
  aiirTransformProducesHandle(resultsVec.data(), resultsVec.size(),
                              effects.effects);
};

void modifiesPayload(PyMemoryEffectsInstanceList effects) {
  aiirTransformModifiesPayload(effects.effects);
}

void onlyReadsPayload(PyMemoryEffectsInstanceList effects) {
  aiirTransformOnlyReadsPayload(effects.effects);
}
} // namespace

static void populateDialectTransformSubmodule(nb::module_ &m) {
  nb::enum_<AiirDiagnosedSilenceableFailure>(m, "DiagnosedSilenceableFailure")
      .value("Success", AiirDiagnosedSilenceableFailureSuccess)
      .value("SilenceableFailure",
             AiirDiagnosedSilenceableFailureSilenceableFailure)
      .value("DefiniteFailure", AiirDiagnosedSilenceableFailureDefiniteFailure);

  AnyOpType::bind(m);
  AnyParamType::bind(m);
  AnyValueType::bind(m);
  OperationType::bind(m);
  ParamType::bind(m);

  PyTransformRewriter::bind(m);
  PyTransformResults::bind(m);
  PyTransformState::bind(m);
  PyTransformOpInterface::bind(m);
  PyPatternDescriptorOpInterface::bind(m);

  m.def("only_reads_handle", onlyReadsHandle,
        "Mark operands as only reading handles.", nb::arg("operands"),
        nb::arg("effects"));

  m.def("consumes_handle", consumesHandle,
        "Mark operands as consuming handles.", nb::arg("operands"),
        nb::arg("effects"));

  m.def("produces_handle", producesHandle, "Mark results as producing handles.",
        nb::arg("results"), nb::arg("effects"));

  m.def("modifies_payload", modifiesPayload,
        "Mark the transform as modifying the payload.", nb::arg("effects"));

  m.def("only_reads_payload", onlyReadsPayload,
        "Mark the transform as only reading the payload.", nb::arg("effects"));
}
} // namespace transform
} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir

NB_MODULE(_aiirDialectsTransform, m) {
  m.doc() = "AIIR Transform dialect.";
  aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN::transform::
      populateDialectTransformSubmodule(m);
}
