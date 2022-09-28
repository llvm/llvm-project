//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_LIBCLANG_CXDIAGNOSTICSETDIAGNOSTICCONSUMER_H
#define LLVM_CLANG_TOOLS_LIBCLANG_CXDIAGNOSTICSETDIAGNOSTICCONSUMER_H

#include "CIndexDiagnostic.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"

namespace clang {
class CXDiagnosticSetDiagnosticConsumer : public DiagnosticConsumer {
  SmallVector<StoredDiagnostic, 4> Errors;

public:
  void HandleDiagnostic(DiagnosticsEngine::Level level,
                        const Diagnostic &Info) override {
    if (level >= DiagnosticsEngine::Error) {
      Errors.push_back(StoredDiagnostic(level, Info));
      ++NumErrors;
    }
  }

  CXDiagnosticSet getDiagnosticSet() {
    return cxdiag::createStoredDiags(Errors, LangOptions());
  }
};
} // namespace clang

#endif
