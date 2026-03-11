//===- MockSummaryViewBuilder1.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MockSummaryViewBuilders.h"
#include "clang/Analysis/Scalable/SummaryView/SummaryViewBuilderRegistry.h"

using namespace clang;
using namespace ssaf;

std::string clang::ssaf::MockBuilderLog;

void clang::ssaf::clearMockBuilderLog() { MockBuilderLog.clear(); }

namespace {

class MockSummaryViewBuilder1
    : public SummaryViewBuilder<MockView1, MockEntitySummary1> {
public:
  MockSummaryViewBuilder1() {
    MockBuilderLog += "MockSummaryViewBuilder1 constructor was invoked\n";
  }

  ~MockSummaryViewBuilder1() {
    MockBuilderLog += "MockSummaryViewBuilder1 destructor was invoked\n";
  }

  void addSummary(EntityId Id, std::unique_ptr<MockEntitySummary1>) override {
    MockBuilderLog += "MockSummaryViewBuilder1 addSummary was invoked\n";
    getView().Ids.push_back(Id);
  }

  void finalize() override {
    MockBuilderLog += "MockSummaryViewBuilder1 finalize was invoked\n";
  }
};

static SummaryViewBuilderRegistry::Add<MockSummaryViewBuilder1>
    Register("Mock1", "Mock view builder 1");

} // namespace
