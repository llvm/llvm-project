//===- MockSummaryViewBuilder2.cpp ----------------------------------------===//
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

namespace {

class MockSummaryViewBuilder2
    : public SummaryViewBuilder<MockView2, MockEntitySummary2> {
public:
  MockSummaryViewBuilder2() {
    MockBuilderLog += "MockSummaryViewBuilder2 constructor was invoked\n";
  }

  ~MockSummaryViewBuilder2() {
    MockBuilderLog += "MockSummaryViewBuilder2 destructor was invoked\n";
  }

  void addSummary(EntityId Id, std::unique_ptr<MockEntitySummary2>) override {
    MockBuilderLog += "MockSummaryViewBuilder2 addSummary was invoked\n";
    getView().Ids.push_back(Id);
  }

  void finalize() override {
    MockBuilderLog += "MockSummaryViewBuilder2 finalize was invoked\n";
  }
};

static SummaryViewBuilderRegistry::Add<MockSummaryViewBuilder2>
    Register("Mock2", "Mock view builder 2");

} // namespace
