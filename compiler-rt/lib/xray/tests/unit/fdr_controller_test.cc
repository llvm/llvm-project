//===-- fdr_controller_test.cc --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a function call tracing system.
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <memory>
#include <time.h>

#include "test_helpers.h"
#include "xray/xray_records.h"
#include "xray_buffer_queue.h"
#include "xray_fdr_controller.h"
#include "xray_fdr_log_writer.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/XRay/Trace.h"
#include "llvm/XRay/XRayRecord.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace __xray {
namespace {

using ::llvm::HasValue;
using ::llvm::xray::testing::FuncId;
using ::llvm::xray::testing::HasArg;
using ::llvm::xray::testing::RecordType;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::SizeIs;

class FunctionSequenceTest : public ::testing::Test {
protected:
  BufferQueue::Buffer B{};
  std::unique_ptr<BufferQueue> BQ;
  std::unique_ptr<FDRLogWriter> W;
  std::unique_ptr<FDRController<>> C;

public:
  void SetUp() override {
    bool Success;
    BQ = llvm::make_unique<BufferQueue>(4096, 1, Success);
    ASSERT_TRUE(Success);
    ASSERT_EQ(BQ->getBuffer(B), BufferQueue::ErrorCode::Ok);
    W = llvm::make_unique<FDRLogWriter>(B);
    C = llvm::make_unique<FDRController<>>(BQ.get(), B, *W, clock_gettime, 0);
  }
};

TEST_F(FunctionSequenceTest, DefaultInitFinalizeFlush) {
  ASSERT_TRUE(C->functionEnter(1, 2, 3));
  ASSERT_TRUE(C->functionExit(1, 2, 3));
  ASSERT_TRUE(C->flush());
  ASSERT_EQ(BQ->finalize(), BufferQueue::ErrorCode::Ok);

  // Serialize the buffers then test to see we find the expected records.
  std::string Serialized = serialize(*BQ, 3);
  llvm::DataExtractor DE(Serialized, true, 8);
  auto TraceOrErr = llvm::xray::loadTrace(DE);
  EXPECT_THAT_EXPECTED(
      TraceOrErr,
      HasValue(ElementsAre(
          AllOf(FuncId(1), RecordType(llvm::xray::RecordTypes::ENTER)),
          AllOf(FuncId(1), RecordType(llvm::xray::RecordTypes::EXIT)))));
}

TEST_F(FunctionSequenceTest, ThresholdsAreEnforced) {
  C = llvm::make_unique<FDRController<>>(BQ.get(), B, *W, clock_gettime, 1000);
  ASSERT_TRUE(C->functionEnter(1, 2, 3));
  ASSERT_TRUE(C->functionExit(1, 2, 3));
  ASSERT_TRUE(C->flush());
  ASSERT_EQ(BQ->finalize(), BufferQueue::ErrorCode::Ok);

  // Serialize the buffers then test to see we find the *no* records, because
  // the function entry-exit comes under the cycle threshold.
  std::string Serialized = serialize(*BQ, 3);
  llvm::DataExtractor DE(Serialized, true, 8);
  auto TraceOrErr = llvm::xray::loadTrace(DE);
  EXPECT_THAT_EXPECTED(TraceOrErr, HasValue(IsEmpty()));
}

TEST_F(FunctionSequenceTest, ArgsAreHandledAndKept) {
  C = llvm::make_unique<FDRController<>>(BQ.get(), B, *W, clock_gettime, 1000);
  ASSERT_TRUE(C->functionEnterArg(1, 2, 3, 4));
  ASSERT_TRUE(C->functionExit(1, 2, 3));
  ASSERT_TRUE(C->flush());
  ASSERT_EQ(BQ->finalize(), BufferQueue::ErrorCode::Ok);

  // Serialize the buffers then test to see we find the function enter arg
  // record with the specified argument.
  std::string Serialized = serialize(*BQ, 3);
  llvm::DataExtractor DE(Serialized, true, 8);
  auto TraceOrErr = llvm::xray::loadTrace(DE);
  EXPECT_THAT_EXPECTED(
      TraceOrErr,
      HasValue(ElementsAre(
          AllOf(FuncId(1), RecordType(llvm::xray::RecordTypes::ENTER_ARG),
                HasArg(4)),
          AllOf(FuncId(1), RecordType(llvm::xray::RecordTypes::EXIT)))));
}

TEST_F(FunctionSequenceTest, RewindingMultipleCalls) {
  C = llvm::make_unique<FDRController<>>(BQ.get(), B, *W, clock_gettime, 1000);

  // First we construct an arbitrarily deep function enter/call stack.
  // We also ensure that we are in the same CPU.
  uint64_t TSC = 1;
  uint16_t CPU = 1;
  ASSERT_TRUE(C->functionEnter(1, TSC++, CPU));
  ASSERT_TRUE(C->functionEnter(2, TSC++, CPU));
  ASSERT_TRUE(C->functionEnter(3, TSC++, CPU));

  // Then we exit them one at a time, in reverse order of entry.
  ASSERT_TRUE(C->functionExit(3, TSC++, CPU));
  ASSERT_TRUE(C->functionExit(2, TSC++, CPU));
  ASSERT_TRUE(C->functionExit(1, TSC++, CPU));

  ASSERT_TRUE(C->flush());
  ASSERT_EQ(BQ->finalize(), BufferQueue::ErrorCode::Ok);

  // Serialize the buffers then test to see we find that all the calls have been
  // unwound because all of them are under the cycle counter threshold.
  std::string Serialized = serialize(*BQ, 3);
  llvm::DataExtractor DE(Serialized, true, 8);
  auto TraceOrErr = llvm::xray::loadTrace(DE);
  EXPECT_THAT_EXPECTED(TraceOrErr, HasValue(IsEmpty()));
}

TEST_F(FunctionSequenceTest, RewindingIntermediaryTailExits) {
  C = llvm::make_unique<FDRController<>>(BQ.get(), B, *W, clock_gettime, 1000);

  // First we construct an arbitrarily deep function enter/call stack.
  // We also ensure that we are in the same CPU.
  uint64_t TSC = 1;
  uint16_t CPU = 1;
  ASSERT_TRUE(C->functionEnter(1, TSC++, CPU));
  ASSERT_TRUE(C->functionEnter(2, TSC++, CPU));
  ASSERT_TRUE(C->functionEnter(3, TSC++, CPU));

  // Next we tail-exit into a new function multiple times.
  ASSERT_TRUE(C->functionTailExit(3, TSC++, CPU));
  ASSERT_TRUE(C->functionEnter(4, TSC++, CPU));
  ASSERT_TRUE(C->functionTailExit(4, TSC++, CPU));
  ASSERT_TRUE(C->functionEnter(5, TSC++, CPU));
  ASSERT_TRUE(C->functionTailExit(5, TSC++, CPU));
  ASSERT_TRUE(C->functionEnter(6, TSC++, CPU));

  // Then we exit them one at a time, in reverse order of entry.
  ASSERT_TRUE(C->functionExit(6, TSC++, CPU));
  ASSERT_TRUE(C->functionExit(2, TSC++, CPU));
  ASSERT_TRUE(C->functionExit(1, TSC++, CPU));
  ASSERT_TRUE(C->flush());
  ASSERT_EQ(BQ->finalize(), BufferQueue::ErrorCode::Ok);

  // Serialize the buffers then test to see we find that all the calls have been
  // unwound because all of them are under the cycle counter threshold.
  std::string Serialized = serialize(*BQ, 3);
  llvm::DataExtractor DE(Serialized, true, 8);
  auto TraceOrErr = llvm::xray::loadTrace(DE);
  EXPECT_THAT_EXPECTED(TraceOrErr, HasValue(IsEmpty()));
}

class BufferManagementTest : public ::testing::Test {
protected:
  BufferQueue::Buffer B{};
  std::unique_ptr<BufferQueue> BQ;
  std::unique_ptr<FDRLogWriter> W;
  std::unique_ptr<FDRController<>> C;

  static constexpr size_t kBuffers = 10;

public:
  void SetUp() override {
    bool Success;
    BQ = llvm::make_unique<BufferQueue>(sizeof(MetadataRecord) * 4 +
                                            sizeof(FunctionRecord) * 2,
                                        kBuffers, Success);
    ASSERT_TRUE(Success);
    ASSERT_EQ(BQ->getBuffer(B), BufferQueue::ErrorCode::Ok);
    W = llvm::make_unique<FDRLogWriter>(B);
    C = llvm::make_unique<FDRController<>>(BQ.get(), B, *W, clock_gettime, 0);
  }
};

constexpr size_t BufferManagementTest::kBuffers;

TEST_F(BufferManagementTest, HandlesOverflow) {
  uint64_t TSC = 1;
  uint16_t CPU = 1;
  for (size_t I = 0; I < kBuffers; ++I) {
    ASSERT_TRUE(C->functionEnter(1, TSC++, CPU));
    ASSERT_TRUE(C->functionExit(1, TSC++, CPU));
  }
  C->flush();
  ASSERT_EQ(BQ->finalize(), BufferQueue::ErrorCode::Ok);

  std::string Serialized = serialize(*BQ, 3);
  llvm::DataExtractor DE(Serialized, true, 8);
  auto TraceOrErr = llvm::xray::loadTrace(DE);
  EXPECT_THAT_EXPECTED(TraceOrErr, HasValue(SizeIs(kBuffers * 2)));
}

TEST_F(BufferManagementTest, HandlesFinalizedBufferQueue) {
  uint64_t TSC = 1;
  uint16_t CPU = 1;

  // First write one function entry.
  ASSERT_TRUE(C->functionEnter(1, TSC++, CPU));

  // Then we finalize the buffer queue, simulating the case where the logging
  // has been finalized.
  ASSERT_EQ(BQ->finalize(), BufferQueue::ErrorCode::Ok);

  // At this point further calls to the controller must fail.
  ASSERT_FALSE(C->functionExit(1, TSC++, CPU));

  // But flushing should succeed.
  ASSERT_TRUE(C->flush());

  // We expect that we'll only be able to find the function enter event, but not
  // the function exit event.
  std::string Serialized = serialize(*BQ, 3);
  llvm::DataExtractor DE(Serialized, true, 8);
  auto TraceOrErr = llvm::xray::loadTrace(DE);
  EXPECT_THAT_EXPECTED(
      TraceOrErr, HasValue(ElementsAre(AllOf(
                      FuncId(1), RecordType(llvm::xray::RecordTypes::ENTER)))));
}

} // namespace
} // namespace __xray
