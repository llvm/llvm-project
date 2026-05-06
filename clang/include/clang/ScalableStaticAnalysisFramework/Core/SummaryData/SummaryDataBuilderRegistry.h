//===- SummaryDataBuilderRegistry.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registry for SummaryDataBuilders.
//
// To register a builder, add a static Add<BuilderT> in the builder's
// translation unit:
//
//   static SummaryDataBuilderRegistry::Add<MyDataBuilder>
//       Registered("Data builder for MyAnalysis");
//
// The registry entry name is derived automatically from
// MyDataBuilder::summaryName(), which returns MyData::summaryName().
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SUMMARYDATA_SUMMARYDATABUILDERREGISTRY_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SUMMARYDATA_SUMMARYDATABUILDERREGISTRY_H

#include "clang/ScalableStaticAnalysisFramework/Core/SummaryData/SummaryDataBuilder.h"
#include "llvm/Support/Registry.h"
#include <memory>
#include <string>

LLVM_DECLARE_REGISTRY(llvm::Registry<clang::ssaf::SummaryDataBuilderBase>)

namespace clang::ssaf {

/// Registry for SummaryDataBuilder implementations.
///
/// Provides an Add helper that derives the registry entry name from
/// BuilderT::summaryName(), eliminating the possibility of registering a
/// builder under the wrong name.
class SummaryDataBuilderRegistry {
  using RegistryT = llvm::Registry<SummaryDataBuilderBase>;

  SummaryDataBuilderRegistry() = delete;

public:
  /// Registers \p BuilderT under the name returned by
  /// \c BuilderT::summaryName(). Only a description is required.
  ///
  /// \c Add objects must be declared \c static at namespace scope — they
  /// register an entry in a global linked list on construction and are
  /// not copyable or movable.
  template <typename BuilderT> struct Add {
    explicit Add(llvm::StringRef Desc)
        : Name(BuilderT::summaryName().str().str()), Node(Name, Desc) {}

    Add(const Add &) = delete;
    Add &operator=(const Add &) = delete;

  private:
    std::string Name;
    RegistryT::Add<BuilderT> Node;
  };

  /// Returns true if a builder is registered under \p Name.
  static bool contains(llvm::StringRef Name);

  /// Instantiates the builder registered under \p Name, or returns nullptr
  /// if no such builder is registered.
  static std::unique_ptr<SummaryDataBuilderBase>
  instantiate(llvm::StringRef Name);
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SUMMARYDATA_SUMMARYDATABUILDERREGISTRY_H
