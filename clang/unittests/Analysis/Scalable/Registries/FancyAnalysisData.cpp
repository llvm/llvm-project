//===- FancyAnalysisData.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Registries/MockSerializationFormat.h"
#include "clang/Analysis/Scalable/TUSummary/EntitySummary.h"
#include "llvm/Support/Registry.h"
#include <vector>

using namespace clang;
using namespace ssaf;

using SpecialFileRepresentation =
    MockSerializationFormat::SpecialFileRepresentation;
using FormatInfo = MockSerializationFormat::FormatInfo;

namespace {
struct FancyAnalysisData : EntitySummary {
  FancyAnalysisData() : EntitySummary(SummaryName("FancyAnalysis")) {}

  std::vector<std::string> SomeInternalList;
};
} // namespace

static SpecialFileRepresentation
serializeFancyAnalysis(const EntitySummary &Data,
                       MockSerializationFormat &Format) {
  const auto &FancyAnalysis = static_cast<const FancyAnalysisData &>(Data);

  std::string Buffer;
  llvm::raw_string_ostream OS(Buffer);
  OS << "FancyAnalysisData{\n";
  OS << "  SomeInternalList: ";
  llvm::interleaveComma(FancyAnalysis.SomeInternalList, OS);
  OS << "\n";
  OS << "}\n";

  return SpecialFileRepresentation{/*MockRepresentation=*/std::move(Buffer)};
}

static std::unique_ptr<EntitySummary>
deserializeFancyAnalysis(const SpecialFileRepresentation &Obj,
                         EntityIdTable &Table) {
  auto Result = std::make_unique<FancyAnalysisData>();

  llvm::StringRef Cursor = Obj.MockRepresentation;
  if (!Cursor.consume_front("FancyAnalysisData{\n  SomeInternalList: "))
    return nullptr;

  auto IsNewLine = [](char C) { return C == '\n'; };
  llvm::StringRef SomeInternalListStr = Cursor.take_until(IsNewLine);
  llvm::SmallVector<llvm::StringRef> Parts;
  SomeInternalListStr.split(Parts, ", ");
  for (llvm::StringRef Part : Parts) {
    Result->SomeInternalList.push_back(Part.str());
  }

  Cursor = Cursor.drop_front(SomeInternalListStr.size());

  if (!Cursor.consume_front("\n}\n"))
    return nullptr;

  if (!Cursor.empty())
    return nullptr;

  return std::move(Result);
}

namespace {
struct FancyAnalysisFormatInfo : FormatInfo {
  FancyAnalysisFormatInfo()
      : FormatInfo{
            SummaryName("FancyAnalysis"),
            serializeFancyAnalysis,
            deserializeFancyAnalysis,
        } {}
};
} // namespace

static llvm::Registry<FormatInfo>::Add<FancyAnalysisFormatInfo>
    RegisterFormatInfo("FancyAnalysisData",
                       "Format info for FancyAnalysisData for the "
                       "MockSerializationFormat1 format");
