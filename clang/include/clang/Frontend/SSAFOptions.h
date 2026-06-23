//===- SSAFOptions.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_SSAFOPTIONS_H
#define LLVM_CLANG_FRONTEND_SSAFOPTIONS_H

#include "llvm/Support/Compiler.h"
#include <string>
#include <vector>

namespace clang::ssaf {

class SSAFOptions {
public:
  /// List of SSAF extractors to enable.
  /// Controlled by: --ssaf-extract-summaries
  std::vector<std::string> ExtractSummaries;

  /// The TU summary output file with the file extension representing the
  /// serialization format.
  /// Controlled by: --ssaf-tu-summary-file
  std::string TUSummaryFile;

  /// Stable identifier used as the name of the `CompilationUnit`
  /// `BuildNamespace` of every produced TU summary.
  /// Controlled by: --ssaf-compilation-unit-id
  std::string CompilationUnitId;

  /// Show the list of available SSAF summary extractors and exit.
  /// Controlled by: --ssaf-list-extractors
  LLVM_PREFERRED_TYPE(bool)
  unsigned ShowExtractors : 1;

  /// Show the list of available SSAF serialization formats and exit.
  /// Controlled by: --ssaf-list-formats
  LLVM_PREFERRED_TYPE(bool)
  unsigned ShowFormats : 1;

  SSAFOptions() {
    ShowExtractors = false;
    ShowFormats = false;
  };
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_FRONTEND_SSAFOPTIONS_H
