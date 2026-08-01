//===- TransformationRegistry.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registry for Transformations, and some helper functions.
// To register a transformation, insert this code:
//
//   namespace clang::ssaf {
//   // NOLINTNEXTLINE(misc-use-internal-linkage)
//   volatile int MyTransformationAnchorSource = 0;
//   } // namespace clang::ssaf
//   static TransformationRegistry::Add<MyTransformation>
//     X("MyTransformation", "My awesome transformation");
//
// For a statically-linked transformation also extend the `AnchorSources`
// list in
// clang/include/clang/ScalableStaticAnalysis/SSAFBuiltinForceLinker.h
// (plugin-loaded transformations do not need an anchor — the dynamic loader
// runs every global ctor on load).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_TRANSFORMATIONREGISTRY_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_TRANSFORMATIONREGISTRY_H

#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/WPASuite.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/SourceEditEmitter.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/Transformation.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/TransformationReportEmitter.h"
#include "clang/Support/Compiler.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Registry.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace clang::ssaf {

/// Check if a Transformation was registered with a given name.
bool isTransformationRegistered(llvm::StringRef Name);

/// Try to instantiate a Transformation with a given name.
/// This might return null if the construction of the desired Transformation
/// failed.
/// It's a fatal error if there is no transformation registered with the name.
std::unique_ptr<Transformation>
makeTransformation(llvm::StringRef Name, const WPASuite &Suite,
                   SourceEditEmitter &Edits,
                   TransformationReportEmitter &Report);

/// Print the list of available Transformations.
void printAvailableTransformations(llvm::raw_ostream &OS);

// Registry for adding new Transformation implementations.
using TransformationRegistry =
    llvm::Registry<Transformation, const WPASuite &, SourceEditEmitter &,
                   TransformationReportEmitter &>;

} // namespace clang::ssaf

LLVM_DECLARE_REGISTRY_EX(CLANG_ABI_EXPORT, clang::ssaf::TransformationRegistry)

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_TRANSFORMATIONREGISTRY_H
