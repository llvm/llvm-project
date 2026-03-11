//===- SummaryViewBuilderRegistry.h ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registry for SummaryViewBuilders.
//
// To register a builder, add a static Add<BuilderT> in the builder's
// translation unit:
//
//   static SummaryViewBuilderRegistry::Add<MyViewBuilder>
//       Registered("View builder for MyAnalysis");
//
// The registry entry name is derived automatically from
// MyViewBuilder::summaryName(), which returns MyView::summaryName().
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYVIEW_SUMMARYVIEWBUILDERREGISTRY_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYVIEW_SUMMARYVIEWBUILDERREGISTRY_H

#include "clang/Analysis/Scalable/SummaryView/SummaryViewBuilder.h"
#include "llvm/Support/Registry.h"
#include <memory>
#include <string>

namespace clang::ssaf {

/// Registry for SummaryViewBuilder implementations.
///
/// Provides an Add helper that derives the registry entry name from
/// BuilderT::summaryName(), eliminating the possibility of registering a
/// builder under the wrong name.
class SummaryViewBuilderRegistry {
  using RegistryT = llvm::Registry<SummaryViewBuilderBase>;

public:
  /// Registers \p BuilderT under the name returned by
  /// \c BuilderT::summaryName(). Only a description is required.
  template <typename BuilderT> struct Add {
    explicit Add(llvm::StringRef Desc)
        : Name(BuilderT::summaryName().str().str()), Node(Name, Desc) {}

  private:
    std::string Name;
    RegistryT::Add<BuilderT> Node;
  };

  /// Returns true if a builder is registered under \p Name.
  static bool isRegistered(llvm::StringRef Name);

  /// Instantiates the builder registered under \p Name, or returns nullptr
  /// if no such builder is registered.
  static std::unique_ptr<SummaryViewBuilderBase>
  instantiate(llvm::StringRef Name);
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_SUMMARYVIEW_SUMMARYVIEWBUILDERREGISTRY_H
