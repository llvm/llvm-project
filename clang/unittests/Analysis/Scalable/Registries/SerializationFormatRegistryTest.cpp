//===- SummaryExtractorRegistryTest.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Serialization/SerializationFormatRegistry.h"
#include "clang/Analysis/Scalable/TUSummary/TUSummary.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <memory>

using namespace llvm;
using namespace clang;
using namespace ssaf;

// Returns the file name and content in a map.
static std::map<std::string, std::string> readFilesFromDir(StringRef DirPath) {
  std::map<std::string, std::string> Result;
  std::error_code EC;

  for (sys::fs::directory_iterator It(DirPath, EC), End; It != End && !EC;
       It.increment(EC)) {
    StringRef FilePath = It->path();

    if (sys::fs::is_directory(FilePath))
      continue;

    auto BufferOrErr = MemoryBuffer::getFile(FilePath);
    if (!BufferOrErr)
      continue;

    // Store only the filename (relative to DirPath).
    StringRef FileName = sys::path::filename(FilePath);
    Result[FileName.str()] = BufferOrErr.get()->getBuffer().str();
  }

  return Result;
}
namespace {

TEST(SerializationFormatRegistryTest, isFormatRegistered) {
  EXPECT_FALSE(isFormatRegistered("Non-existent-format"));
  EXPECT_TRUE(isFormatRegistered("MockSerializationFormat"));
}

TEST(SerializationFormatRegistryTest, EnumeratingRegistryEntries) {
  auto Formats = SerializationFormatRegistry::entries();
  ASSERT_EQ(std::distance(Formats.begin(), Formats.end()), 1U);
  EXPECT_EQ(Formats.begin()->getName(), "MockSerializationFormat");
}

TEST(SerializationFormatRegistryTest, Roundtrip) {
  // Create temporary input directory
  SmallString<128> InputDir;
  std::error_code EC = sys::fs::createUniqueDirectory("ssaf-input", InputDir);
  ASSERT_FALSE(EC) << "Failed to create input directory: " << EC.message();
  llvm::scope_exit CleanupInputOnExit(
      [&] { sys::fs::remove_directories(InputDir); });

  // Create input files
  SmallString<128> AnalysesFile = InputDir;
  sys::path::append(AnalysesFile, "analyses.txt");
  {
    raw_fd_ostream OS(AnalysesFile, EC);
    ASSERT_FALSE(EC) << "Failed to create analyses.txt: " << EC.message();
    OS << "FancyAnalysis\n";
  }

  SmallString<128> FancyAnalysisFile = InputDir;
  sys::path::append(FancyAnalysisFile, "FancyAnalysis.special");
  {
    raw_fd_ostream OS(FancyAnalysisFile, EC);
    ASSERT_FALSE(EC) << "Failed to create FancyAnalysis.special: "
                     << EC.message();
    OS << "Some FancyAnalysisData...";
  }

  std::unique_ptr<SerializationFormat> Format =
      makeFormat("MockSerializationFormat");
  ASSERT_TRUE(Format);

  auto LoadedSummaryOrErr = Format->readTUSummary(InputDir);
  ASSERT_THAT_EXPECTED(LoadedSummaryOrErr, Succeeded());
  TUSummary LoadedSummary = std::move(*LoadedSummaryOrErr);

  // Create a temporary output directory
  SmallString<128> OutputDir;
  EC = sys::fs::createUniqueDirectory("ssaf-test", OutputDir);
  ASSERT_FALSE(EC) << "Failed to create temporary directory: " << EC.message();
  llvm::scope_exit CleanupOnExit(
      [&] { sys::fs::remove_directories(OutputDir); });

  auto WriteErr = Format->writeTUSummary(LoadedSummary, OutputDir);
  ASSERT_THAT_ERROR(std::move(WriteErr), Succeeded());

  EXPECT_EQ(readFilesFromDir(OutputDir),
            (std::map<std::string, std::string>{
                {"analyses.txt", "FancyAnalysis\n"},
                {"FancyAnalysis.special", "Some FancyAnalysisData..."},
            }));
}

} // namespace
