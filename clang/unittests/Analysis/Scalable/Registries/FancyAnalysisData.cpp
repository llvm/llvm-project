//===- FancyAnalysisData.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Registries/MockSerializationFormat.h"
#include "clang/Analysis/Scalable/TUSummary/EntitySummary.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Registry.h"

using namespace clang;
using namespace ssaf;

using SpecialFileRepresentation =
    MockSerializationFormat::SpecialFileRepresentation;

namespace {
struct FancyAnalysisData final
    : llvm::RTTIExtends<FancyAnalysisData, EntitySummary> {
  SummaryName getSummaryName() const override {
    return SummaryName("FancyAnalysis");
  }

  std::string Text;
  static char ID;
};
char FancyAnalysisData::ID = 0;
} // namespace

static SpecialFileRepresentation
serializeFancyAnalysis(const EntitySummary &Data,
                       MockSerializationFormat &Format) {
  const auto &FancyAnalysis = llvm::cast<FancyAnalysisData>(Data);
  return SpecialFileRepresentation{/*MockRepresentation=*/FancyAnalysis.Text};
}

static std::unique_ptr<EntitySummary>
deserializeFancyAnalysis(const SpecialFileRepresentation &File,
                         EntityIdTable &) {
  auto Result = std::make_unique<FancyAnalysisData>();
  Result->Text = File.MockRepresentation;
  return std::move(Result);
}

namespace {
using FormatInfo = MockSerializationFormat::FormatInfo;
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
                       "MockSerializationFormat format");
