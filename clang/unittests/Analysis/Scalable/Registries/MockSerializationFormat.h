//===- MockSerializationFormat.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_ANALYSIS_SCALABLE_REGISTRIES_MOCKSERIALIZATIONFORMAT_H
#define LLVM_CLANG_UNITTESTS_ANALYSIS_SCALABLE_REGISTRIES_MOCKSERIALIZATIONFORMAT_H

#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include "clang/Analysis/Scalable/Serialization/SerializationFormat.h"
#include "clang/Support/Compiler.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Registry.h"
#include <string>

namespace clang::ssaf {

class MockSerializationFormat final : public SerializationFormat {
public:
  MockSerializationFormat();

  llvm::Expected<TUSummary> readTUSummary(llvm::StringRef Path) override;

  llvm::Error writeTUSummary(const TUSummary &Summary,
                             llvm::StringRef Path) override;

  struct SpecialFileRepresentation {
    std::string MockRepresentation;
  };

  using SerializerFn = llvm::function_ref<SpecialFileRepresentation(
      const EntitySummary &, MockSerializationFormat &)>;
  using DeserializerFn = llvm::function_ref<std::unique_ptr<EntitySummary>(
      const SpecialFileRepresentation &, EntityIdTable &)>;

  using FormatInfo = FormatInfoEntry<SerializerFn, DeserializerFn>;
  std::map<SummaryName, FormatInfo> FormatInfos;

  static char ID;
};

} // namespace clang::ssaf

namespace llvm {
extern template class CLANG_TEMPLATE_ABI
    Registry<clang::ssaf::MockSerializationFormat::FormatInfo>;
} // namespace llvm

#endif // LLVM_CLANG_UNITTESTS_ANALYSIS_SCALABLE_REGISTRIES_MOCKSERIALIZATIONFORMAT_H
