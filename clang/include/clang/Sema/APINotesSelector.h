//===--- APINotesSelector.h - API Notes selector helpers --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_APINOTESSELECTOR_H
#define LLVM_CLANG_SEMA_APINOTESSELECTOR_H

#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <string>

namespace clang {

class ASTContext;
class FunctionDecl;

using APINotesParameterSelector = llvm::SmallVector<std::string, 4>;

struct APINotesParameterSelectorCandidates {
  APINotesParameterSelector Source;
  std::optional<APINotesParameterSelector> Desugared;
};

std::optional<APINotesParameterSelectorCandidates>
getAPINotesParameterSelectorCandidates(const ASTContext &Context,
                                       const FunctionDecl *FD);

} // namespace clang

#endif // LLVM_CLANG_SEMA_APINOTESSELECTOR_H
