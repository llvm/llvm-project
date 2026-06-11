//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYACTIONCONTROLLER_H
#define LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYACTIONCONTROLLER_H

#include <memory>
#include <optional>
#include <string>

namespace clang {

class CompilerInstance;
class CompilerInvocation;
class CowCompilerInvocation;

namespace dependencies {
struct ModuleDeps;

/// An output from a module compilation, such as the path of the module file.
enum class ModuleOutputKind {
  /// The module file (.pcm). Required.
  ModuleFile,
  /// The path of the dependency file (.d), if any.
  DependencyFile,
  /// The null-separated list of names to use as the targets in the dependency
  /// file, if any. Defaults to the value of \c ModuleFile, as in the driver.
  DependencyTargets,
  /// The path of the serialized diagnostic file (.dia), if any.
  DiagnosticSerializationFile,
};

/// Dependency scanner callbacks that are used during scanning to influence the
/// behaviour of the scan - for example, to customize the scanned invocations.
class DependencyActionController {
public:
  virtual ~DependencyActionController() = default;

  /// Creates a copy of the controller. The result must be both thread-safe.
  virtual std::unique_ptr<DependencyActionController> clone() const = 0;

  /// Provides output path for a given module dependency. Must be thread-safe.
  virtual std::string lookupModuleOutput(const ModuleDeps &MD,
                                         ModuleOutputKind Kind) = 0;

  /// Initializes the scan invocation.
  virtual void initializeScanInvocation(CompilerInvocation &ScanInvocation) {}

  /// Initializes the scan instance and modifies the resulting TU invocation.
  /// Returns true on success, false on failure.
  virtual bool initialize(CompilerInstance &ScanInstance,
                          CompilerInvocation &NewInvocation) {
    return true;
  }

  /// Finalizes the scan instance and modifies the resulting TU invocation.
  /// Returns true on success, false on failure.
  virtual bool finalize(CompilerInstance &ScanInstance,
                        CompilerInvocation &NewInvocation) {
    return true;
  }

  /// Returns the cache key for the resulting invocation, or nullopt.
  virtual std::optional<std::string>
  getCacheKey(const CompilerInvocation &NewInvocation) {
    return std::nullopt;
  }

  /// Initializes the module scan instance.
  /// Returns true on success, false on failure.
  virtual bool initializeModuleBuild(CompilerInstance &ModuleScanInstance) {
    return true;
  }

  /// Finalizes the module scan instance.
  /// Returns true on success, false on failure.
  virtual bool finalizeModuleBuild(CompilerInstance &ModuleScanInstance) {
    return true;
  }

  /// Modifies the resulting module invocation and the associated structure.
  /// Returns true on success, false on failure.
  virtual bool finalizeModuleInvocation(CompilerInstance &ScanInstance,
                                        CowCompilerInvocation &CI,
                                        const ModuleDeps &MD) {
    return true;
  }
};
} // namespace dependencies
} // namespace clang

#endif // LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYACTIONCONTROLLER_H
