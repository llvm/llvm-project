//===- SourcePassAnalysisRegistry.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registry for SourcePassAnalysis subclasses.
//
// To register an analysis, add a static Add<AnalysisT> and an anchor source
// in its translation unit, then add the matching anchor destination to the
// relevant force-linker header:
//
//   // MySourcePassAnalysis.cpp
//   static SourcePassAnalysisRegistry::Add<MySourcePassAnalysis>
//       Registered("One-line description of MySourcePassAnalysis");
//
//   volatile int SSAFMySourcePassAnalysisAnchorSource = 0;
//
//   // SSAFBuiltinForceLinker.h (or the relevant force-linker header)
//   extern volatile int SSAFMySourcePassAnalysisAnchorSource;
//   [[maybe_unused]] static int SSAFMySourcePassAnalysisAnchorDestination =
//       SSAFMySourcePassAnalysisAnchorSource;
//
// The registry entry name is derived automatically from
// MySourcePassAnalysis::analysisName(), so name-mismatch bugs are impossible.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SOURCEPASSANALYSIS_SOURCEPASSANALYSISREGISTRY_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SOURCEPASSANALYSIS_SOURCEPASSANALYSISREGISTRY_H

#include "clang/ScalableStaticAnalysisFramework/Core/SourcePassAnalysis/SourcePassAnalysis.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Support/ErrorBuilder.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/WPASuite.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Registry.h"
#include <memory>
#include <string>
#include <vector>

namespace clang::ssaf {
/// Registry for SourcePassAnalysis implementations.
class SourcePassAnalysisRegistry {
  SourcePassAnalysisRegistry() = delete;

public:
  using RegistryT =
      llvm::Registry<SourcePassAnalysisBase, std::unique_ptr<WPASuite>>;

  /// Registers AnalysisT with the registry.
  ///
  /// The registry entry name is derived automatically from
  /// AnalysisT::ResultType::analysisName(), so name-mismatch bugs are
  /// impossible.
  ///
  /// Add objects must be declared static at namespace scope.
  template <class AnalysisT, typename ResultT, typename DepResultT> struct Add {
    static_assert(
        std::is_base_of_v<SourcePassAnalysis<ResultT, DepResultT>, AnalysisT>,
        "AnalysisT must derive from SourcePassAnalysis<...>");

    explicit Add(llvm::StringRef Desc)
        : Name(AnalysisT::analysisName().str().str()), Node(Name, Desc) {
      if (contains(AnalysisT::analysisName())) {
        ErrorBuilder::fatal("duplicate analysis registration for '{0}'", Name);
      }
      getAnalysisNames().push_back(AnalysisT::analysisName());
    }

    Add(const Add &) = delete;
    Add &operator=(const Add &) = delete;

  private:
    std::string Name;
    RegistryT::Add<AnalysisT> Node;
  };

  /// Returns true if an analysis is registered under \p Name.
  static bool contains(const AnalysisName &Name);

  /// Returns the names of all registered analyses.
  static const std::vector<AnalysisName> &names();

  /// Instantiates the analysis registered under \p Name, or returns an error
  /// if no such analysis is registered.
  static llvm::Expected<std::unique_ptr<SourcePassAnalysisBase>>
  instantiate(const AnalysisName &Name, std::unique_ptr<WPASuite> WPAResult);

private:
  /// Returns the global list of registered analysis names.
  ///
  /// Uses a function-local static to avoid static initialization order
  /// fiasco: Add<T> objects in other translation units may push names before
  /// a plain static data member could be constructed.
  static std::vector<AnalysisName> &getAnalysisNames();
};

} // namespace clang::ssaf

LLVM_DECLARE_REGISTRY(clang::ssaf::SourcePassAnalysisRegistry::RegistryT)

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SOURCEPASSANALYSIS_SOURCEPASSANALYSISREGISTRY_H
