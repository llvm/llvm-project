//===- MockSerializationFormat.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Registries/MockSerializationFormat.h"
#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "clang/Analysis/Scalable/Model/EntityName.h"
#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include "clang/Analysis/Scalable/Serialization/SerializationFormat.h"
#include "clang/Analysis/Scalable/Serialization/SerializationFormatRegistry.h"
#include "clang/Analysis/Scalable/TUSummary/TUSummary.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <functional>
#include <memory>
#include <set>

using namespace clang;
using namespace ssaf;

LLVM_INSTANTIATE_REGISTRY(llvm::Registry<MockSerializationFormat::FormatInfo>)

MockSerializationFormat::MockSerializationFormat() {
  for (const auto &FormatInfoEntry : llvm::Registry<FormatInfo>::entries()) {
    std::unique_ptr<FormatInfo> Info = FormatInfoEntry.instantiate();
    bool Inserted = FormatInfos.try_emplace(Info->ForSummary, *Info).second;
    if (!Inserted) {
      llvm::report_fatal_error(
          "Format info was already registered for summary name: " +
          Info->ForSummary.str());
    }
  }
}

llvm::Expected<TUSummary>
MockSerializationFormat::readTUSummary(llvm::StringRef Path) {
  BuildNamespace NS(BuildNamespaceKind::CompilationUnit, "Mock.cpp");
  TUSummary Summary(NS);

  auto ManifestFile = llvm::MemoryBuffer::getFile(Path + "/analyses.txt");
  if (!ManifestFile) {
    return llvm::createStringError(ManifestFile.getError(),
                                   "Failed to read manifest file");
  }
  llvm::StringRef ManifestFileContent = (*ManifestFile)->getBuffer();

  llvm::SmallVector<llvm::StringRef, 5> Analyses;
  ManifestFileContent.split(Analyses, /*Separator=*/"\n", /*MaxSplit=*/-1,
                            /*KeepEmpty=*/false);

  for (llvm::StringRef Analysis : Analyses) {
    SummaryName Name(Analysis.str());
    auto InputFile =
        llvm::MemoryBuffer::getFile(Path + "/" + Name.str() + ".special");
    if (!InputFile) {
      return llvm::createStringError(InputFile.getError(),
                                     "Failed to read analysis file");
    }
    auto InfoIt = FormatInfos.find(Name);
    if (InfoIt == FormatInfos.end()) {
      return llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "No FormatInfo was registered for summary name: " + Name.str());
    }
    const auto &InfoEntry = InfoIt->second;
    assert(InfoEntry.ForSummary == Name);

    SpecialFileRepresentation Repr{(*InputFile)->getBuffer().str()};
    auto &Table = getIdTable(Summary);

    std::unique_ptr<EntitySummary> Result = InfoEntry.Deserialize(Repr, Table);
    if (!Result) {
      return llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "Failed to deserialize EntitySummary for analysis: " + Name.str());
    }

    EntityId FooId = Table.getId(EntityName{"c:@F@foo", "", /*Namespace=*/{}});
    auto &IdMappings = getData(Summary).try_emplace(Name).first->second;
    [[maybe_unused]] bool Inserted =
        IdMappings.try_emplace(FooId, std::move(Result)).second;
    assert(Inserted);
  }

  return std::move(Summary);
}

llvm::Error MockSerializationFormat::writeTUSummary(const TUSummary &Summary,
                                                    llvm::StringRef Path) {
  std::error_code EC;

  // Check if output directory exists, create if needed
  if (!llvm::sys::fs::exists(Path)) {
    EC = llvm::sys::fs::create_directories(Path);
    if (EC) {
      return llvm::createStringError(EC, "Failed to create output directory '" +
                                             Path + "': " + EC.message());
    }
  }

  std::set<SummaryName> Analyses;
  for (const auto &[SummaryName, EntityMappings] : getData(Summary)) {
    [[maybe_unused]] bool Inserted = Analyses.insert(SummaryName).second;
    assert(Inserted);
    for (const auto &Data : llvm::make_second_range(EntityMappings)) {
      auto InfoIt = FormatInfos.find(SummaryName);
      if (InfoIt == FormatInfos.end()) {
        return llvm::createStringError(
            std::make_error_code(std::errc::invalid_argument),
            "There was no FormatInfo registered for summary name '" +
                SummaryName.str() + "'");
      }
      const auto &InfoEntry = InfoIt->second;
      assert(InfoEntry.ForSummary == SummaryName);

      auto Output = InfoEntry.Serialize(*Data, *this);

      std::string AnalysisFilePath =
          (Path + "/" + SummaryName.str() + ".special").str();
      llvm::raw_fd_ostream AnalysisOutputFile(AnalysisFilePath, EC);
      if (EC) {
        return llvm::createStringError(
            EC, "Failed to create file '" + AnalysisFilePath +
                    "': " + llvm::StringRef(EC.message()));
      }
      AnalysisOutputFile << Output.MockRepresentation;
    }
  }

  std::string ManifestFilePath = (Path + "/analyses.txt").str();
  llvm::raw_fd_ostream ManifestFile(ManifestFilePath, EC);
  if (EC) {
    return llvm::createStringError(
        EC, "Failed to create manifest file '" + ManifestFilePath +
                "': " + llvm::StringRef(EC.message()));
  }

  interleave(map_range(Analyses, std::mem_fn(&SummaryName::str)), ManifestFile,
             "\n");
  ManifestFile << "\n";

  return llvm::Error::success();
}

static SerializationFormatRegistry::Add<MockSerializationFormat>
    RegisterFormat("MockSerializationFormat",
                   "A serialization format for testing");
