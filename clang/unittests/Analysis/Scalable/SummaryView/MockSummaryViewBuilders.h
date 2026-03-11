//===- MockSummaryViewBuilders.h
//-------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared mock types for SummaryView tests. Two mock analyses ("Mock1" and
// "Mock2") are defined here, with their builders registered in
// MockSummaryViewBuilder1.cpp and MockSummaryViewBuilder2.cpp.
//
// Tests observe builder behaviour via MockBuilderLog, which records lifecycle
// events (constructor, addSummary, finalize, destructor) as newline-terminated
// strings. Call clearMockBuilderLog() in SetUp() to isolate tests.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_UNITTESTS_ANALYSIS_SCALABLE_SUMMARYVIEW_MOCKSUMMARYVIEWBUILDERS_H
#define CLANG_UNITTESTS_ANALYSIS_SCALABLE_SUMMARYVIEW_MOCKSUMMARYVIEWBUILDERS_H

#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include "clang/Analysis/Scalable/SummaryView/SummaryView.h"
#include "clang/Analysis/Scalable/SummaryView/SummaryViewBuilder.h"
#include "clang/Analysis/Scalable/TUSummary/EntitySummary.h"
#include <string>
#include <vector>

namespace clang::ssaf {

// ---- Mock entity summaries -----------------------------------------------

class MockEntitySummary1 : public EntitySummary {
public:
  SummaryName getSummaryName() const override { return SummaryName("Mock1"); }
};

class MockEntitySummary2 : public EntitySummary {
public:
  SummaryName getSummaryName() const override { return SummaryName("Mock2"); }
};

// ---- Mock views ------------------------------------------------------------

class MockView1 : public SummaryView {
public:
  static SummaryName summaryName() { return SummaryName("Mock1"); }
  std::vector<EntityId> Ids;
};

class MockView2 : public SummaryView {
public:
  static SummaryName summaryName() { return SummaryName("Mock2"); }
  std::vector<EntityId> Ids;
};

// ---- Shared log ------------------------------------------------------------
// Defined in MockSummaryViewBuilder1.cpp; written by both mock builders.

extern std::string MockBuilderLog;

void clearMockBuilderLog();

} // namespace clang::ssaf

#endif // CLANG_UNITTESTS_ANALYSIS_SCALABLE_SUMMARYVIEW_MOCKSUMMARYVIEWBUILDERS_H
