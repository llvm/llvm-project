//===- CASDependencyCollector.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_CASDEPENDENCYCOLLECTOR_H
#define LLVM_CLANG_FRONTEND_CASDEPENDENCYCOLLECTOR_H

#include "clang/Frontend/Utils.h"
#include "llvm/CAS/CASReference.h"

namespace clang {

/// Collects dependencies when attached to a Preprocessor (for includes) and
/// ASTReader (for module imports), and writes it to the CAS in a manner
/// suitable to be replayed into a DependencyFileGenerator.
class CASDependencyCollector : public DependencyFileGenerator {
public:
  /// Create a \CASDependencyCollector for the given output options.
  ///
  /// \param Opts Output options. Only options that affect the list of
  ///             dependency files are significant.
  /// \param CAS The CAS to write the dependency list to.
  /// \param Callback Callback that receives the resulting dependencies on
  ///                 completion, or \c None if an error occurred.
  CASDependencyCollector(
      DependencyOutputOptions Opts, cas::ObjectStore &CAS,
      std::function<void(Optional<cas::ObjectRef>)> Callback);

  /// Replay the given result, which should have been created by a
  /// \c CASDependencyCollector instance.
  ///
  /// \param Opts Output options. Only options that affect the output format of
  ///             a dependency file are signficant.
  /// \param CAS The CAS to read the result from.
  /// \param DepsRef The dependencies.
  /// \param OS The output stream to write the dependency file to.
  static llvm::Error replay(const DependencyOutputOptions &Opts,
                            cas::ObjectStore &CAS, cas::ObjectRef DepsRef,
                            llvm::raw_ostream &OS);

private:
  void finishedMainFile(DiagnosticsEngine &Diags) override;

  cas::ObjectStore &CAS;
  std::function<void(Optional<cas::ObjectRef>)> Callback;
};

} // namespace clang

#endif // LLVM_CLANG_FRONTEND_CASDEPENDENCYCOLLECTOR_H
