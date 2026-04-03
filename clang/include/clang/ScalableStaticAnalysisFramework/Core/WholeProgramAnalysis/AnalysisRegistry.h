//===- AnalysisRegistry.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unified registry for both SummaryAnalysis and DerivedAnalysis subclasses.
//
// To register an analysis, add a static Add<AnalysisT> and an anchor source
// in its translation unit, then add the matching anchor destination to the
// relevant force-linker header:
//
//   // MyAnalysis.cpp
//   static AnalysisRegistry::Add<MyAnalysis>
//       Registered("One-line description of MyAnalysis");
//
//   volatile int SSAFMyAnalysisAnchorSource = 0;
//
//   // SSAFBuiltinForceLinker.h (or the relevant force-linker header)
//   extern volatile int SSAFMyAnalysisAnchorSource;
//   [[maybe_unused]] static int SSAFMyAnalysisAnchorDestination =
//       SSAFMyAnalysisAnchorSource;
//
// The registry entry name is derived automatically from
// MyAnalysis::analysisName(), so name-mismatch bugs are impossible.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_WHOLEPROGRAMANALYSIS_ANALYSISREGISTRY_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_WHOLEPROGRAMANALYSIS_ANALYSISREGISTRY_H

#include "clang/ScalableStaticAnalysisFramework/Core/Support/ErrorBuilder.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/DerivedAnalysis.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/SummaryAnalysis.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Registry.h"
#include <memory>
#include <string>
#include <vector>

namespace clang::ssaf {

/// Unified registry for SummaryAnalysis and DerivedAnalysis implementations.
///
/// Internally uses a single llvm::Registry<AnalysisBase>. The correct kind
/// is carried by the AnalysisBase::TheKind tag set in each subclass
/// constructor.
class AnalysisRegistry {
  using RegistryT = llvm::Registry<AnalysisBase>;

  AnalysisRegistry() = delete;

public:
  /// Registers AnalysisT with the unified registry.
  ///
  /// The registry entry name is derived automatically from
  /// AnalysisT::ResultType::analysisName(), so name-mismatch bugs are
  /// impossible.
  ///
  /// Add objects must be declared static at namespace scope.
  template <typename AnalysisT> struct Add {
    static_assert(std::is_base_of_v<SummaryAnalysisBase, AnalysisT> ||
                      std::is_base_of_v<DerivedAnalysisBase, AnalysisT>,
                  "AnalysisT must derive from SummaryAnalysis<...> or "
                  "DerivedAnalysis<...>");

    explicit Add(llvm::StringRef Desc)
        : Name(AnalysisT::ResultType::analysisName().str().str()),
          Node(Name, Desc) {
      if (contains(AnalysisT::ResultType::analysisName())) {
        ErrorBuilder::fatal("duplicate analysis registration for '{0}'", Name);
      }
      getAnalysisNames().push_back(AnalysisT::ResultType::analysisName());
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
  static llvm::Expected<std::unique_ptr<AnalysisBase>>
  instantiate(const AnalysisName &Name);

private:
  /// Returns the global list of registered analysis names.
  ///
  /// Uses a function-local static to avoid static initialization order
  /// fiasco: Add<T> objects in other translation units may push names before
  /// a plain static data member could be constructed.
  static std::vector<AnalysisName> &getAnalysisNames();
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_WHOLEPROGRAMANALYSIS_ANALYSISREGISTRY_H
