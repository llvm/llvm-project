//=== UnsignedStatDemo.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This checker demonstrates the use of UnsignedEPStat for per-entry-point
// statistics. It conditionally sets a statistic based on the entry point name.
//
//===----------------------------------------------------------------------===//

#include "CheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/EntryPointStats.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"
#include <optional>

using namespace clang;
using namespace ento;

static UnsignedEPStat DemoStat("DemoStat");

namespace {
class UnsignedStatTesterChecker : public Checker<check::BeginFunction> {
public:
  void checkBeginFunction(CheckerContext &C) const {
    StringRef Name;
    if (const Decl *D = C.getLocationContext()->getDecl())
      if (const FunctionDecl *F = D->getAsFunction())
        Name = F->getName();

    // Conditionally set the statistic based on the function name (leaving it
    // undefined for all other functions)
    if (Name == "func_one")
      DemoStat.set(1);
    else if (Name == "func_two")
      DemoStat.set(2);
    else
      ; // For any other function (e.g., "func_none") don't set the statistic
  }
};

void addUnsignedStatTesterChecker(AnalysisASTConsumer &AnalysisConsumer,
                                  AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"test.DemoStatChecker", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker<UnsignedStatTesterChecker>(
        "test.DemoStatChecker", "DescriptionOfDemoStatChecker");
  });
}

// Find the index of a column in the CSV header.
// Returns std::nullopt if the column is not found.
static std::optional<unsigned>
findColumnIndex(llvm::ArrayRef<llvm::StringRef> Header,
                llvm::StringRef ColumnName) {
  auto Iter = llvm::find(Header, ColumnName);
  if (Iter != Header.end())
    return std::distance(Header.begin(), Iter);
  return std::nullopt;
}

// Parse CSV content and extract a mapping from one column to another.
// KeyColumn is used as the map key (e.g., "DebugName").
// ValueColumn is used as the map value (e.g., "DemoStat").
// Returns a map from key column values to value column values.
static llvm::StringMap<std::string>
parseCSVColumnMapping(llvm::StringRef CSVContent, llvm::StringRef KeyColumn,
                      llvm::StringRef ValueColumn) {
  llvm::StringMap<std::string> Result;

  // Parse CSV: first line is header, subsequent lines are data
  llvm::SmallVector<llvm::StringRef, 8> Lines;
  CSVContent.split(Lines, '\n', -1, false);
  if (Lines.size() < 2) // Need at least header + one data row
    return Result;

  // Parse header to find column indices
  llvm::SmallVector<llvm::StringRef, 32> Header;
  Lines[0].split(Header, ',');
  std::optional<unsigned> KeyIdx = findColumnIndex(Header, KeyColumn);
  std::optional<unsigned> ValueIdx = findColumnIndex(Header, ValueColumn);

  if (!KeyIdx || !ValueIdx)
    return Result;

  // Parse data rows and extract mappings
  for (auto Line : llvm::drop_begin(Lines)) {
    llvm::SmallVector<llvm::StringRef, 32> Row;
    Line.split(Row, ',');
    if (Row.size() <= std::max(*KeyIdx, *ValueIdx))
      continue;

    llvm::StringRef KeyVal = Row[*KeyIdx].trim().trim('"');
    llvm::StringRef ValueVal = Row[*ValueIdx].trim().trim('"');

    if (!KeyVal.empty())
      Result[KeyVal] = ValueVal.str();
  }

  return Result;
}

TEST(UnsignedStat, ExplicitlySetUnsignedStatistic) {
  llvm::SmallString<128> TempMetricsCsvPath;
  std::error_code EC =
      llvm::sys::fs::createTemporaryFile("ep_stats", "csv", TempMetricsCsvPath);
  ASSERT_FALSE(EC);
  std::vector<std::string> Args = {
      "-Xclang", "-analyzer-config", "-Xclang",
      std::string("dump-entry-point-stats-to-csv=") +
          TempMetricsCsvPath.str().str()};
  // Clean up on exit
  auto Cleanup = llvm::make_scope_exit(
      [&]() { llvm::sys::fs::remove(TempMetricsCsvPath); });
  EXPECT_TRUE(runCheckerOnCodeWithArgs<addUnsignedStatTesterChecker>(
      R"cpp(
            void func_one() {}
            void func_two() {}
            void func_none() {}
      )cpp",
      Args));

  auto BufferOrError = llvm::MemoryBuffer::getFile(TempMetricsCsvPath);
  ASSERT_TRUE(BufferOrError);
  llvm::StringRef CSVContent = BufferOrError.get()->getBuffer();

  // Parse the CSV and extract function statistics
  llvm::StringMap<std::string> FunctionStats =
      parseCSVColumnMapping(CSVContent, "DebugName", "DemoStat");

  // Verify the expected values
  ASSERT_TRUE(FunctionStats.count("func_one()"));
  EXPECT_EQ(FunctionStats["func_one()"], "1");

  ASSERT_TRUE(FunctionStats.count("func_two()"));
  EXPECT_EQ(FunctionStats["func_two()"], "2");

  ASSERT_TRUE(FunctionStats.count("func_none()"));
  EXPECT_EQ(FunctionStats["func_none()"], ""); // Not set, should be empty
}
} // namespace
