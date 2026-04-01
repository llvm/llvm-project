//===- Pass.cpp - Pass Management -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Pass.h"

#include "aiir-c/Pass.h"
#include "aiir/Bindings/Python/Globals.h"
#include "aiir/Bindings/Python/IRCore.h"
// clang-format off
#include "aiir/Bindings/Python/Nanobind.h"
#include "aiir-c/Bindings/Python/Interop.h" // This is expected after nanobind.
// clang-format on

namespace nb = nanobind;
using namespace nb::literals;
using namespace aiir;
using namespace aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN;

namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {

/// Owning Wrapper around a PassManager.
class PyPassManager {
public:
  PyPassManager(AiirPassManager passManager) : passManager(passManager) {}
  PyPassManager(PyPassManager &&other) noexcept
      : passManager(other.passManager) {
    other.passManager.ptr = nullptr;
  }
  ~PyPassManager() {
    if (!aiirPassManagerIsNull(passManager))
      aiirPassManagerDestroy(passManager);
  }
  AiirPassManager get() { return passManager; }

  void release() { passManager.ptr = nullptr; }
  nb::object getCapsule() {
    return nb::steal<nb::object>(aiirPythonPassManagerToCapsule(get()));
  }

  static nb::object createFromCapsule(const nb::object &capsule) {
    AiirPassManager rawPm = aiirPythonCapsuleToPassManager(capsule.ptr());
    if (aiirPassManagerIsNull(rawPm))
      throw nb::python_error();
    return nb::cast(PyPassManager(rawPm), nb::rv_policy::move);
  }

private:
  AiirPassManager passManager;
};

enum class PyAiirPassDisplayMode : std::underlying_type_t<AiirPassDisplayMode> {
  LIST = AIIR_PASS_DISPLAY_MODE_LIST,
  PIPELINE = AIIR_PASS_DISPLAY_MODE_PIPELINE
};

struct PyAiirExternalPass : AiirExternalPass {};

/// Create the `aiir.passmanager` here.
void populatePassManagerSubmodule(nb::module_ &m) {
  //----------------------------------------------------------------------------
  // Mapping of enumerated types
  //----------------------------------------------------------------------------
  nb::enum_<PyAiirPassDisplayMode>(m, "PassDisplayMode")
      .value("LIST", PyAiirPassDisplayMode::LIST)
      .value("PIPELINE", PyAiirPassDisplayMode::PIPELINE);

  //----------------------------------------------------------------------------
  // Mapping of AiirExternalPass
  //----------------------------------------------------------------------------
  nb::class_<PyAiirExternalPass>(m, "ExternalPass")
      .def("signal_pass_failure", [](PyAiirExternalPass pass) {
        aiirExternalPassSignalFailure(pass);
      });

  //----------------------------------------------------------------------------
  // Mapping of the top-level PassManager
  //----------------------------------------------------------------------------
  nb::class_<PyPassManager>(m, "PassManager")
      .def(
          "__init__",
          [](PyPassManager &self, const std::string &anchorOp,
             DefaultingPyAiirContext context) {
            AiirPassManager passManager = aiirPassManagerCreateOnOperation(
                context->get(),
                aiirStringRefCreate(anchorOp.data(), anchorOp.size()));
            new (&self) PyPassManager(passManager);
          },
          "anchor_op"_a = nb::str("any"), "context"_a = nb::none(),
          // clang-format off
          nb::sig("def __init__(self, anchor_op: str = 'any', context: " MAKE_AIIR_PYTHON_QUALNAME("ir.Context") " | None = None) -> None"),
          // clang-format on
          "Create a new PassManager for the current (or provided) Context.")
      .def_prop_ro(AIIR_PYTHON_CAPI_PTR_ATTR, &PyPassManager::getCapsule)
      .def(AIIR_PYTHON_CAPI_FACTORY_ATTR, &PyPassManager::createFromCapsule)
      .def("_testing_release", &PyPassManager::release,
           "Releases (leaks) the backing pass manager (testing)")
      .def(
          "enable_ir_printing",
          [](PyPassManager &passManager, bool printBeforeAll,
             bool printAfterAll, bool printModuleScope, bool printAfterChange,
             bool printAfterFailure, std::optional<int64_t> largeElementsLimit,
             std::optional<int64_t> largeResourceLimit, bool enableDebugInfo,
             bool printGenericOpForm,
             std::optional<std::string> optionalTreePrintingPath) {
            AiirOpPrintingFlags flags = aiirOpPrintingFlagsCreate();
            if (largeElementsLimit) {
              aiirOpPrintingFlagsElideLargeElementsAttrs(flags,
                                                         *largeElementsLimit);
              aiirOpPrintingFlagsElideLargeResourceString(flags,
                                                          *largeElementsLimit);
            }
            if (largeResourceLimit)
              aiirOpPrintingFlagsElideLargeResourceString(flags,
                                                          *largeResourceLimit);
            if (enableDebugInfo)
              aiirOpPrintingFlagsEnableDebugInfo(flags, /*enable=*/true,
                                                 /*prettyForm=*/false);
            if (printGenericOpForm)
              aiirOpPrintingFlagsPrintGenericOpForm(flags);
            std::string treePrintingPath = "";
            if (optionalTreePrintingPath.has_value())
              treePrintingPath = optionalTreePrintingPath.value();
            aiirPassManagerEnableIRPrinting(
                passManager.get(), printBeforeAll, printAfterAll,
                printModuleScope, printAfterChange, printAfterFailure, flags,
                aiirStringRefCreate(treePrintingPath.data(),
                                    treePrintingPath.size()));
            aiirOpPrintingFlagsDestroy(flags);
          },
          "print_before_all"_a = false, "print_after_all"_a = true,
          "print_module_scope"_a = false, "print_after_change"_a = false,
          "print_after_failure"_a = false,
          "large_elements_limit"_a = nb::none(),
          "large_resource_limit"_a = nb::none(), "enable_debug_info"_a = false,
          "print_generic_op_form"_a = false,
          "tree_printing_dir_path"_a = nb::none(),
          "Enable IR printing, default as aiir-print-ir-after-all.")
      .def(
          "enable_verifier",
          [](PyPassManager &passManager, bool enable) {
            aiirPassManagerEnableVerifier(passManager.get(), enable);
          },
          "enable"_a, "Enable / disable verify-each.")
      .def(
          "enable_timing",
          [](PyPassManager &passManager) {
            aiirPassManagerEnableTiming(passManager.get());
          },
          "Enable pass timing.")
      .def(
          "enable_statistics",
          [](PyPassManager &passManager, PyAiirPassDisplayMode displayMode) {
            aiirPassManagerEnableStatistics(
                passManager.get(),
                static_cast<AiirPassDisplayMode>(displayMode));
          },
          "displayMode"_a = PyAiirPassDisplayMode::PIPELINE,
          "Enable pass statistics.")
      .def_static(
          "parse",
          [](const std::string &pipeline, DefaultingPyAiirContext context) {
            AiirPassManager passManager = aiirPassManagerCreate(context->get());
            PyPrintAccumulator errorMsg;
            AiirLogicalResult status = aiirParsePassPipeline(
                aiirPassManagerGetAsOpPassManager(passManager),
                aiirStringRefCreate(pipeline.data(), pipeline.size()),
                errorMsg.getCallback(), errorMsg.getUserData());
            if (aiirLogicalResultIsFailure(status))
              throw nb::value_error(errorMsg.join().c_str());
            return new PyPassManager(passManager);
          },
          "pipeline"_a, "context"_a = nb::none(),
          // clang-format off
          nb::sig("def parse(pipeline: str, context: " MAKE_AIIR_PYTHON_QUALNAME("ir.Context") " | None = None) -> PassManager"),
          // clang-format on
          "Parse a textual pass-pipeline and return a top-level PassManager "
          "that can be applied on a Module. Throw a ValueError if the pipeline "
          "can't be parsed")
      .def(
          "add",
          [](PyPassManager &passManager, const std::string &pipeline) {
            PyPrintAccumulator errorMsg;
            AiirLogicalResult status = aiirOpPassManagerAddPipeline(
                aiirPassManagerGetAsOpPassManager(passManager.get()),
                aiirStringRefCreate(pipeline.data(), pipeline.size()),
                errorMsg.getCallback(), errorMsg.getUserData());
            if (aiirLogicalResultIsFailure(status))
              throw nb::value_error(errorMsg.join().c_str());
          },
          "pipeline"_a,
          "Add textual pipeline elements to the pass manager. Throws a "
          "ValueError if the pipeline can't be parsed.")
      .def(
          "add",
          [](PyPassManager &passManager, const nb::callable &run,
             std::optional<std::string> &name, const std::string &argument,
             const std::string &description, const std::string &opName) {
            if (!name.has_value()) {
              name = nb::cast<std::string>(
                  nb::borrow<nb::str>(run.attr("__name__")));
            }
            AiirTypeID passID = PyGlobals::get().allocateTypeID();
            AiirExternalPassCallbacks callbacks;
            callbacks.construct = [](void *obj) {
              (void)nb::handle(static_cast<PyObject *>(obj)).inc_ref();
            };
            callbacks.destruct = [](void *obj) {
              (void)nb::handle(static_cast<PyObject *>(obj)).dec_ref();
            };
            callbacks.initialize = nullptr;
            callbacks.clone = [](void *) -> void * {
              throw std::runtime_error("Cloning Python passes not supported");
            };
            callbacks.run = [](AiirOperation op, AiirExternalPass pass,
                               void *userData) {
              nb::handle(static_cast<PyObject *>(userData))(
                  op, PyAiirExternalPass{pass.ptr});
            };
            auto externalPass = aiirCreateExternalPass(
                passID, aiirStringRefCreate(name->data(), name->length()),
                aiirStringRefCreate(argument.data(), argument.length()),
                aiirStringRefCreate(description.data(), description.length()),
                aiirStringRefCreate(opName.data(), opName.size()),
                /*nDependentDialects*/ 0, /*dependentDialects*/ nullptr,
                callbacks, /*userData*/ run.ptr());
            aiirPassManagerAddOwnedPass(passManager.get(), externalPass);
          },
          "run"_a, "name"_a.none() = nb::none(), "argument"_a.none() = "",
          "description"_a.none() = "", "op_name"_a.none() = "",
          R"(
            Add a python-defined pass to the current pipeline of the pass manager.

            Args:
              run: A callable with signature ``(op: ir.Operation, pass_: ExternalPass) -> None``.
                   Called when the pass executes. It receives the operation to be processed and
                   the current ``ExternalPass`` instance.
                   Use ``pass_.signal_pass_failure()`` to signal failure.
              name: The name of the pass. Defaults to ``run.__name__``.
              argument: The command-line argument for the pass. Defaults to empty.
              description: The description of the pass. Defaults to empty.
              op_name: The name of the operation this pass operates on.
                       It will be a generic operation pass if not specified.)")
      .def(
          "run",
          [](PyPassManager &passManager, PyOperationBase &op) {
            // Actually run the pass manager.
            PyAiirContext::ErrorCapture errors(op.getOperation().getContext());
            AiirLogicalResult status = aiirPassManagerRunOnOp(
                passManager.get(), op.getOperation().get());
            if (aiirLogicalResultIsFailure(status))
              throw AIIRError("Failure while executing pass pipeline",
                              errors.take());
          },
          "operation"_a,
          // clang-format off
          nb::sig("def run(self, operation: " MAKE_AIIR_PYTHON_QUALNAME("ir._OperationBase") ") -> None"),
          // clang-format on
          "Run the pass manager on the provided operation, raising an "
          "AIIRError on failure.")
      .def(
          "__str__",
          [](PyPassManager &self) {
            AiirPassManager passManager = self.get();
            PyPrintAccumulator printAccum;
            aiirPrintPassPipeline(
                aiirPassManagerGetAsOpPassManager(passManager),
                printAccum.getCallback(), printAccum.getUserData());
            return printAccum.join();
          },
          "Print the textual representation for this PassManager, suitable to "
          "be passed to `parse` for round-tripping.");
}
} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir
