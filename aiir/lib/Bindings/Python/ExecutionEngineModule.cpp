//===- ExecutionEngineModule.cpp - Python module for execution engine -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <vector>

#include "aiir-c/ExecutionEngine.h"
#include "aiir/Bindings/Python/IRCore.h"
#include "aiir/Bindings/Python/Nanobind.h"
#include "aiir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {
namespace execution_engine {

/// Owning Wrapper around an ExecutionEngine.
class PyExecutionEngine {
public:
  PyExecutionEngine(AiirExecutionEngine executionEngine)
      : executionEngine(executionEngine) {}
  PyExecutionEngine(PyExecutionEngine &&other) noexcept
      : executionEngine(other.executionEngine) {
    other.executionEngine.ptr = nullptr;
  }
  ~PyExecutionEngine() {
    if (!aiirExecutionEngineIsNull(executionEngine))
      aiirExecutionEngineDestroy(executionEngine);
  }
  AiirExecutionEngine get() { return executionEngine; }

  void release() {
    executionEngine.ptr = nullptr;
    referencedObjects.clear();
  }
  nb::object getCapsule() {
    return nb::steal<nb::object>(aiirPythonExecutionEngineToCapsule(get()));
  }

  // Add an object to the list of referenced objects whose lifetime must exceed
  // those of the ExecutionEngine.
  void addReferencedObject(const nb::object &obj) {
    referencedObjects.push_back(obj);
  }

  static nb::object createFromCapsule(const nb::object &capsule) {
    AiirExecutionEngine rawPm =
        aiirPythonCapsuleToExecutionEngine(capsule.ptr());
    if (aiirExecutionEngineIsNull(rawPm))
      throw nb::python_error();
    return nb::cast(PyExecutionEngine(rawPm), nb::rv_policy::move);
  }

private:
  AiirExecutionEngine executionEngine;
  // We support Python ctypes closures as callbacks. Keep a list of the objects
  // so that they don't get garbage collected. (The ExecutionEngine itself
  // just holds raw pointers with no lifetime semantics).
  std::vector<nb::object> referencedObjects;
};

} // namespace execution_engine
} // namespace AIIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace aiir

/// Create the `aiir.execution_engine` module here.
NB_MODULE(_aiirExecutionEngine, m) {
  m.doc() = "AIIR Execution Engine";

  using namespace aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN;
  using namespace execution_engine;
  //----------------------------------------------------------------------------
  // Mapping of the top-level PassManager
  //----------------------------------------------------------------------------
  nb::class_<PyExecutionEngine>(m, "ExecutionEngine")
      .def(
          "__init__",
          [](PyExecutionEngine &self, PyModule &module, int optLevel,
             const std::vector<std::string> &sharedLibPaths,
             bool enableObjectDump, bool enablePIC) {
            std::vector<AiirStringRef> libPaths;
            libPaths.reserve(sharedLibPaths.size());
            for (const std::string &path : sharedLibPaths)
              libPaths.push_back({path.c_str(), path.length()});
            AiirExecutionEngine executionEngine = aiirExecutionEngineCreate(
                module.get(), optLevel, libPaths.size(), libPaths.data(),
                enableObjectDump, enablePIC);
            if (aiirExecutionEngineIsNull(executionEngine))
              throw std::runtime_error(
                  "Failure while creating the ExecutionEngine.");
            new (&self) PyExecutionEngine(executionEngine);
          },
          nb::arg("module"), nb::arg("opt_level") = 2,
          nb::arg("shared_libs") = nb::list(),
          nb::arg("enable_object_dump") = true, nb::arg("enable_pic") = false,
          "Create a new ExecutionEngine instance for the given Module. The "
          "module must contain only dialects that can be translated to LLVM. "
          "Perform transformations and code generation at the optimization "
          "level `opt_level` if specified, or otherwise at the default "
          "level of two (-O2). Load a list of libraries specified in "
          "`shared_libs`.")
      .def_prop_ro(AIIR_PYTHON_CAPI_PTR_ATTR, &PyExecutionEngine::getCapsule)
      .def("_testing_release", &PyExecutionEngine::release,
           "Releases (leaks) the backing ExecutionEngine (for testing purpose)")
      .def(AIIR_PYTHON_CAPI_FACTORY_ATTR, &PyExecutionEngine::createFromCapsule)
      .def(
          "raw_lookup",
          [](PyExecutionEngine &executionEngine, const std::string &func) {
            auto *res = aiirExecutionEngineLookupPacked(
                executionEngine.get(),
                aiirStringRefCreate(func.c_str(), func.size()));
            return reinterpret_cast<uintptr_t>(res);
          },
          nb::arg("func_name"),
          "Lookup function `func` in the ExecutionEngine.")
      .def(
          "raw_register_runtime",
          [](PyExecutionEngine &executionEngine, const std::string &name,
             const nb::object &callbackObj) {
            executionEngine.addReferencedObject(callbackObj);
            uintptr_t rawSym =
                nb::cast<uintptr_t>(nb::getattr(callbackObj, "value"));
            aiirExecutionEngineRegisterSymbol(
                executionEngine.get(),
                aiirStringRefCreate(name.c_str(), name.size()),
                reinterpret_cast<void *>(rawSym));
          },
          nb::arg("name"), nb::arg("callback"),
          "Register `callback` as the runtime symbol `name`.")
      .def(
          "initialize",
          [](PyExecutionEngine &executionEngine) {
            aiirExecutionEngineInitialize(executionEngine.get());
          },
          "Initialize the ExecutionEngine. Global constructors specified by "
          "`llvm.aiir.global_ctors` will be run. One common scenario is that "
          "kernel binary compiled from `gpu.module` gets loaded during "
          "initialization. Make sure all symbols are resolvable before "
          "initialization by calling `register_runtime` or including "
          "shared libraries.")
      .def(
          "dump_to_object_file",
          [](PyExecutionEngine &executionEngine, const std::string &fileName) {
            aiirExecutionEngineDumpToObjectFile(
                executionEngine.get(),
                aiirStringRefCreate(fileName.c_str(), fileName.size()));
          },
          nb::arg("file_name"), "Dump ExecutionEngine to an object file.");
}
