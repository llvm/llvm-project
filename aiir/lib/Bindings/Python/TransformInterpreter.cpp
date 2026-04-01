//===- TransformInterpreter.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pybind classes for the transform dialect interpreter.
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/Transform/Interpreter.h"
#include "aiir-c/IR.h"
#include "aiir-c/Support.h"
#include "aiir/Bindings/Python/Diagnostics.h"
#include "aiir/Bindings/Python/IRCore.h"
#include "aiir/Bindings/Python/Nanobind.h"
#include "aiir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {
namespace transform_interpreter {
struct PyTransformOptions {
  PyTransformOptions() { options = aiirTransformOptionsCreate(); };
  PyTransformOptions(PyTransformOptions &&other) {
    options = other.options;
    other.options.ptr = nullptr;
  }
  PyTransformOptions(const PyTransformOptions &) = delete;

  ~PyTransformOptions() { aiirTransformOptionsDestroy(options); }

  AiirTransformOptions options;
};
} // namespace transform_interpreter
} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir

static void populateTransformInterpreterSubmodule(nb::module_ &m) {
  using namespace aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN;
  using namespace transform_interpreter;
  nb::class_<PyTransformOptions>(m, "TransformOptions")
      .def(nb::init<>())
      .def_prop_rw(
          "expensive_checks",
          [](const PyTransformOptions &self) {
            return aiirTransformOptionsGetExpensiveChecksEnabled(self.options);
          },
          [](PyTransformOptions &self, bool value) {
            aiirTransformOptionsEnableExpensiveChecks(self.options, value);
          })
      .def_prop_rw(
          "enforce_single_top_level_transform_op",
          [](const PyTransformOptions &self) {
            return aiirTransformOptionsGetEnforceSingleTopLevelTransformOp(
                self.options);
          },
          [](PyTransformOptions &self, bool value) {
            aiirTransformOptionsEnforceSingleTopLevelTransformOp(self.options,
                                                                 value);
          });

  m.def(
      "apply_named_sequence",
      [](PyOperationBase &payloadRoot, PyOperationBase &transformRoot,
         PyOperationBase &transformModule, const PyTransformOptions &options) {
        aiir::python::CollectDiagnosticsToStringScope scope(
            aiirOperationGetContext(transformRoot.getOperation()));
        AiirLogicalResult result = aiirTransformApplyNamedSequence(
            payloadRoot.getOperation(), transformRoot.getOperation(),
            transformModule.getOperation(), options.options);
        if (aiirLogicalResultIsSuccess(result)) {
          // Even in cases of success, we might have diagnostics to report:
          std::string msg;
          if ((msg = scope.takeMessage()).size() > 0) {
            fprintf(stderr,
                    "Diagnostic generated while applying "
                    "transform.named_sequence:\n%s",
                    msg.data());
          }
          return;
        }

        throw nb::value_error(
            ("Failed to apply named transform sequence.\nDiagnostic message " +
             scope.takeMessage())
                .c_str());
      },
      nb::arg("payload_root"), nb::arg("transform_root"),
      nb::arg("transform_module"),
      nb::arg("transform_options") = PyTransformOptions());

  m.def(
      "copy_symbols_and_merge_into",
      [](PyOperationBase &target, PyOperationBase &other) {
        aiir::python::CollectDiagnosticsToStringScope scope(
            aiirOperationGetContext(target.getOperation()));

        AiirLogicalResult result = aiirMergeSymbolsIntoFromClone(
            target.getOperation(), other.getOperation());
        if (aiirLogicalResultIsFailure(result)) {
          throw nb::value_error(
              ("Failed to merge symbols.\nDiagnostic message " +
               scope.takeMessage())
                  .c_str());
        }
      },
      nb::arg("target"), nb::arg("other"));
}

NB_MODULE(_aiirTransformInterpreter, m) {
  m.doc() = "AIIR Transform dialect interpreter functionality.";
  populateTransformInterpreterSubmodule(m);
}
