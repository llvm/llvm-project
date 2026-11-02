//===- MockSerializationFormat.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSIS_REGISTRIES_MOCKSERIALIZATIONFORMAT_H
#define LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSIS_REGISTRIES_MOCKSERIALIZATIONFORMAT_H

#include "clang/ScalableStaticAnalysis/Core/Model/SummaryName.h"
#include "clang/ScalableStaticAnalysis/Core/Serialization/SerializationFormat.h"
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

  llvm::Expected<TUSummaryEncoding>
  readTUSummaryEncoding(llvm::StringRef Path) override;

  llvm::Error writeTUSummaryEncoding(const TUSummaryEncoding &SummaryEncoding,
                                     llvm::StringRef Path) override;

  llvm::Expected<LUSummary> readLUSummary(llvm::StringRef Path) override;

  llvm::Error writeLUSummary(const LUSummary &Summary,
                             llvm::StringRef Path) override;

  llvm::Expected<LUSummaryEncoding>
  readLUSummaryEncoding(llvm::StringRef Path) override;

  llvm::Error writeLUSummaryEncoding(const LUSummaryEncoding &SummaryEncoding,
                                     llvm::StringRef Path) override;

  llvm::Expected<StaticLibrary>
  readStaticLibrary(llvm::StringRef Path) override;

  llvm::Error writeStaticLibrary(const StaticLibrary &S,
                                 llvm::StringRef Path) override;

  llvm::Expected<MultiArchStaticLibrary>
  readMultiArchStaticLibrary(llvm::StringRef Path) override;

  llvm::Error writeMultiArchStaticLibrary(const MultiArchStaticLibrary &M,
                                          llvm::StringRef Path) override;

  llvm::Expected<MultiArchSharedLibrary>
  readMultiArchSharedLibrary(llvm::StringRef Path) override;

  llvm::Error writeMultiArchSharedLibrary(const MultiArchSharedLibrary &M,
                                          llvm::StringRef Path) override;

  llvm::Expected<Artifact> readArtifact(llvm::StringRef Path) override;

  llvm::Error writeArtifact(const Artifact &A, llvm::StringRef Path) override;

  llvm::Expected<ArtifactEncoding>
  readArtifactEncoding(llvm::StringRef Path) override;

  llvm::Error writeArtifactEncoding(const ArtifactEncoding &E,
                                    llvm::StringRef Path) override;

  llvm::Expected<WPASuite> readWPASuite(llvm::StringRef Path) override;

  llvm::Error writeWPASuite(const WPASuite &Suite,
                            llvm::StringRef Path) override;

  /// Lists what analyses implement this particular serialisation format.
  void forEachRegisteredAnalysis(
      llvm::function_ref<void(llvm::StringRef Name, llvm::StringRef Desc)>
          Callback) const override;

  struct SpecialFileRepresentation {
    std::string MockRepresentation;
  };

  using SerializerFn = llvm::function_ref<SpecialFileRepresentation(
      const EntitySummary &, MockSerializationFormat &)>;
  using DeserializerFn = llvm::function_ref<std::unique_ptr<EntitySummary>(
      const SpecialFileRepresentation &, EntityIdTable &)>;

  using FormatInfo = FormatInfoEntry<SerializerFn, DeserializerFn>;
  std::map<SummaryName, FormatInfo> FormatInfos;
};

} // namespace clang::ssaf

LLVM_DECLARE_REGISTRY(
    llvm::Registry<clang::ssaf::MockSerializationFormat::FormatInfo>)

#endif // LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSIS_REGISTRIES_MOCKSERIALIZATIONFORMAT_H
