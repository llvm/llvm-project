//===-- TraceTests.cpp - Tracing unit tests ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Trace.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/YAMLParser.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

MATCHER_P(StringNode, Val, "") {
  if (arg->getType() != llvm::yaml::Node::NK_Scalar) {
    *result_listener << "is a " << arg->getVerbatimTag();
    return false;
  }
  llvm::SmallString<32> S;
  return Val == static_cast<llvm::yaml::ScalarNode *>(arg)->getValue(S);
}

// Checks that N is a Mapping (JS object) with the expected scalar properties.
// The object must have all the Expected properties, but may have others.
bool VerifyObject(llvm::yaml::Node &N,
                  std::map<std::string, std::string> Expected) {
  auto *M = llvm::dyn_cast<llvm::yaml::MappingNode>(&N);
  if (!M) {
    ADD_FAILURE() << "Not an object";
    return false;
  }
  bool Match = true;
  llvm::SmallString<32> Tmp;
  for (auto &Prop : *M) {
    auto *K = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(Prop.getKey());
    if (!K)
      continue;
    std::string KS = K->getValue(Tmp).str();
    auto I = Expected.find(KS);
    if (I == Expected.end())
      continue; // Ignore properties with no assertion.

    auto *V = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(Prop.getValue());
    if (!V) {
      ADD_FAILURE() << KS << " is not a string";
      Match = false;
    }
    std::string VS = V->getValue(Tmp).str();
    if (VS != I->second) {
      ADD_FAILURE() << KS << " expected " << I->second << " but actual " << VS;
      Match = false;
    }
    Expected.erase(I);
  }
  for (const auto &P : Expected) {
    ADD_FAILURE() << P.first << " missing, expected " << P.second;
    Match = false;
  }
  return Match;
}

TEST(TraceTest, SmokeTest) {
  // Capture some events.
  std::string JSON;
  {
    llvm::raw_string_ostream OS(JSON);
    auto JSONTracer = trace::createJSONTracer(OS);
    trace::Session Session(*JSONTracer);
    {
      trace::Span Tracer("A");
      trace::log("B");
    }
  }

  // Get the root JSON object using the YAML parser.
  llvm::SourceMgr SM;
  llvm::yaml::Stream Stream(JSON, SM);
  auto Doc = Stream.begin();
  ASSERT_NE(Doc, Stream.end());
  auto *Root = llvm::dyn_cast_or_null<llvm::yaml::MappingNode>(Doc->getRoot());
  ASSERT_NE(Root, nullptr) << "Root should be an object";

  // Check whether we expect thread name events on this platform.
  llvm::SmallString<32> ThreadName;
  get_thread_name(ThreadName);
  bool ThreadsHaveNames = !ThreadName.empty();

  // We expect in order:
  //   displayTimeUnit: "ns"
  //   traceEvents: [process name, thread name, start span, log, end span]
  // (The order doesn't matter, but the YAML parser is awkward to use otherwise)
  auto Prop = Root->begin();
  ASSERT_NE(Prop, Root->end()) << "Expected displayTimeUnit property";
  ASSERT_THAT(Prop->getKey(), StringNode("displayTimeUnit"));
  EXPECT_THAT(Prop->getValue(), StringNode("ns"));
  ASSERT_NE(++Prop, Root->end()) << "Expected traceEvents property";
  EXPECT_THAT(Prop->getKey(), StringNode("traceEvents"));
  auto *Events =
      llvm::dyn_cast_or_null<llvm::yaml::SequenceNode>(Prop->getValue());
  ASSERT_NE(Events, nullptr) << "traceEvents should be an array";
  auto Event = Events->begin();
  ASSERT_NE(Event, Events->end()) << "Expected process name";
  EXPECT_TRUE(VerifyObject(*Event, {{"ph", "M"}, {"name", "process_name"}}));
  if (ThreadsHaveNames) {
    ASSERT_NE(++Event, Events->end()) << "Expected thread name";
    EXPECT_TRUE(VerifyObject(*Event, {{"ph", "M"}, {"name", "thread_name"}}));
  }
  ASSERT_NE(++Event, Events->end()) << "Expected log message";
  EXPECT_TRUE(VerifyObject(*Event, {{"ph", "i"}, {"name", "Log"}}));
  ASSERT_NE(++Event, Events->end()) << "Expected span end";
  EXPECT_TRUE(VerifyObject(*Event, {{"ph", "X"}, {"name", "A"}}));
  ASSERT_EQ(++Event, Events->end());
  ASSERT_EQ(++Prop, Root->end());
}

} // namespace
} // namespace clangd
} // namespace clang
