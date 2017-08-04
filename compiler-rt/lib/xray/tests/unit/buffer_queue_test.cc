//===-- buffer_queue_test.cc ----------------------------------------------===//
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
#include "xray_buffer_queue.h"
#include "gtest/gtest.h"

#include <future>
#include <unistd.h>

namespace __xray {

static constexpr size_t kSize = 4096;

TEST(BufferQueueTest, API) {
  bool Success = false;
  BufferQueue Buffers(kSize, 1, Success);
  ASSERT_TRUE(Success);
}

TEST(BufferQueueTest, GetAndRelease) {
  bool Success = false;
  BufferQueue Buffers(kSize, 1, Success);
  ASSERT_TRUE(Success);
  BufferQueue::Buffer Buf;
  ASSERT_EQ(Buffers.getBuffer(Buf), BufferQueue::ErrorCode::Ok);
  ASSERT_NE(nullptr, Buf.Buffer);
  ASSERT_EQ(Buffers.releaseBuffer(Buf), BufferQueue::ErrorCode::Ok);
  ASSERT_EQ(nullptr, Buf.Buffer);
}

TEST(BufferQueueTest, GetUntilFailed) {
  bool Success = false;
  BufferQueue Buffers(kSize, 1, Success);
  ASSERT_TRUE(Success);
  BufferQueue::Buffer Buf0;
  EXPECT_EQ(Buffers.getBuffer(Buf0), BufferQueue::ErrorCode::Ok);
  BufferQueue::Buffer Buf1;
  EXPECT_EQ(BufferQueue::ErrorCode::NotEnoughMemory, Buffers.getBuffer(Buf1));
  EXPECT_EQ(Buffers.releaseBuffer(Buf0), BufferQueue::ErrorCode::Ok);
}

TEST(BufferQueueTest, ReleaseUnknown) {
  bool Success = false;
  BufferQueue Buffers(kSize, 1, Success);
  ASSERT_TRUE(Success);
  BufferQueue::Buffer Buf;
  Buf.Buffer = reinterpret_cast<void *>(0xdeadbeef);
  Buf.Size = kSize;
  EXPECT_EQ(BufferQueue::ErrorCode::UnrecognizedBuffer,
            Buffers.releaseBuffer(Buf));
}

TEST(BufferQueueTest, ErrorsWhenFinalising) {
  bool Success = false;
  BufferQueue Buffers(kSize, 2, Success);
  ASSERT_TRUE(Success);
  BufferQueue::Buffer Buf;
  ASSERT_EQ(Buffers.getBuffer(Buf), BufferQueue::ErrorCode::Ok);
  ASSERT_NE(nullptr, Buf.Buffer);
  ASSERT_EQ(Buffers.finalize(), BufferQueue::ErrorCode::Ok);
  BufferQueue::Buffer OtherBuf;
  ASSERT_EQ(BufferQueue::ErrorCode::AlreadyFinalized,
            Buffers.getBuffer(OtherBuf));
  ASSERT_EQ(BufferQueue::ErrorCode::AlreadyFinalized,
            Buffers.finalize());
  ASSERT_EQ(Buffers.releaseBuffer(Buf), BufferQueue::ErrorCode::Ok);
}

TEST(BufferQueueTest, MultiThreaded) {
  bool Success = false;
  BufferQueue Buffers(kSize, 100, Success);
  ASSERT_TRUE(Success);
  auto F = [&] {
    BufferQueue::Buffer B;
    while (true) {
      auto EC = Buffers.getBuffer(B);
      if (EC != BufferQueue::ErrorCode::Ok)
        return;
      Buffers.releaseBuffer(B);
    }
  };
  auto T0 = std::async(std::launch::async, F);
  auto T1 = std::async(std::launch::async, F);
  auto T2 = std::async(std::launch::async, [&] {
    while (Buffers.finalize() != BufferQueue::ErrorCode::Ok)
      ;
  });
  F();
}

TEST(BufferQueueTest, Apply) {
  bool Success = false;
  BufferQueue Buffers(kSize, 10, Success);
  ASSERT_TRUE(Success);
  auto Count = 0;
  BufferQueue::Buffer B;
  for (int I = 0; I < 10; ++I) {
    ASSERT_EQ(Buffers.getBuffer(B), BufferQueue::ErrorCode::Ok);
    ASSERT_EQ(Buffers.releaseBuffer(B), BufferQueue::ErrorCode::Ok);
  }
  Buffers.apply([&](const BufferQueue::Buffer &B) { ++Count; });
  ASSERT_EQ(Count, 10);
}

} // namespace __xray
