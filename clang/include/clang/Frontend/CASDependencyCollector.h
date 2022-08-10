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

namespace llvm::cas {
class CASOutputBackend;
}

namespace clang {

/// Collects dependencies when attached to a Preprocessor (for includes) and
/// ASTReader (for module imports), and writes it to the CAS in a manner
/// suitable to be replayed into a DependencyFileGenerator.
class CASDependencyCollector : public DependencyFileGenerator {
public:
  CASDependencyCollector(
      const DependencyOutputOptions &Opts,
      IntrusiveRefCntPtr<llvm::cas::CASOutputBackend> OutputBackend);

  static llvm::Error replay(const DependencyOutputOptions &Opts,
                            cas::CASDB &CAS, cas::ObjectRef DepsRef,
                            llvm::raw_ostream &OS);

private:
  void finishedMainFile(DiagnosticsEngine &Diags) override;

  IntrusiveRefCntPtr<llvm::cas::CASOutputBackend> CASOutputs;
  std::string OutputName;
};

} // namespace clang

#endif // LLVM_CLANG_FRONTEND_CASDEPENDENCYCOLLECTOR_H