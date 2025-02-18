//===-- SymbolDocumentationMatchers.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// GMock matchers for the SymbolDocumentation class
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SYMBOLDOCUMENTATION_MATCHERS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SYMBOLDOCUMENTATION_MATCHERS_H
#include "SymbolDocumentation.h"
#include "gmock/gmock.h"

namespace clang {
namespace clangd {

template <class S>
testing::Matcher<SymbolDocumentation<S>>
matchesDoc(const SymbolDocumentation<S> &Expected) {
  using namespace ::testing;

  std::vector<Matcher<ParameterDocumentation<S>>> ParamMatchers;
  for (const auto &P : Expected.Parameters)
    ParamMatchers.push_back(
        AllOf(Field("Name", &ParameterDocumentation<S>::Name, P.Name),
              Field("Description", &ParameterDocumentation<S>::Description,
                    P.Description)));

  return AllOf(
      Field("Brief", &SymbolDocumentation<S>::Brief, Expected.Brief),
      Field("Returns", &SymbolDocumentation<S>::Returns, Expected.Returns),
      Field("Notes", &SymbolDocumentation<S>::Notes,
            ElementsAreArray(Expected.Notes)),
      Field("Warnings", &SymbolDocumentation<S>::Warnings,
            ElementsAreArray(Expected.Warnings)),
      Field("Parameters", &SymbolDocumentation<S>::Parameters,
            ElementsAreArray(ParamMatchers)),
      Field("Description", &SymbolDocumentation<S>::Description,
            Expected.Description),
      Field("CommentText", &SymbolDocumentation<S>::CommentText,
            Expected.CommentText));
}

} // namespace clangd
} // namespace clang

#endif
