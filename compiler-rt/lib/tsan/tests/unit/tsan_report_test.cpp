//===-- tsan_report_test.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// Tests for the DescribeX functions in tsan_report.cpp, which produce the
// uncolored, newline-free description strings shared with the PrintX path.
//
// NOTE: The exact string format asserted in these tests is not a stable
// contract. These checks just pin down the current output so the DescribeX /
// PrintX split stays in sync; if the report wording changes, update the
// expected strings here to match.
//
//===----------------------------------------------------------------------===//
#include "tsan_report.h"

#include "gtest/gtest.h"

namespace __tsan {

TEST(Report, DescribeMop) {
  ReportMop mop;
  mop.tid = 1;
  mop.addr = 0x1234;
  mop.size = 4;
  mop.write = true;
  mop.atomic = false;
  mop.external_tag = kExternalTagNone;

  InternalScopedString s;
  DescribeMop(&mop, /*first=*/true, s);
  EXPECT_STREQ("  Write of size 4 at 0x1234 by thread T1", s.data());

  s.clear();
  mop.write = false;
  DescribeMop(&mop, /*first=*/false, s);
  EXPECT_STREQ("  Previous read of size 4 at 0x1234 by thread T1", s.data());

  s.clear();
  mop.write = true;
  mop.atomic = true;
  mop.tid = kMainTid;
  DescribeMop(&mop, /*first=*/true, s);
  EXPECT_STREQ("  Atomic write of size 4 at 0x1234 by main thread", s.data());
}

TEST(Report, DescribeLocationStackAndTls) {
  ReportLocation loc;
  loc.tid = 2;

  loc.type = ReportLocationStack;
  InternalScopedString s;
  // Stack/TLS locations carry no following stack trace.
  EXPECT_FALSE(DescribeLocation(&loc, s));
  EXPECT_STREQ("  Location is stack of thread T2.", s.data());

  loc.type = ReportLocationTLS;
  s.clear();
  EXPECT_FALSE(DescribeLocation(&loc, s));
  EXPECT_STREQ("  Location is TLS of thread T2.", s.data());
}

TEST(Report, DescribeLocationHeap) {
  ReportLocation loc;
  loc.type = ReportLocationHeap;
  loc.tid = 3;
  loc.heap_chunk_start = 0x4000;
  loc.heap_chunk_size = 16;
  loc.external_tag = kExternalTagNone;

  InternalScopedString s;
  // Heap locations are followed by the allocation stack.
  EXPECT_TRUE(DescribeLocation(&loc, s));
  EXPECT_STREQ(
      "  Location is heap block of size 16 at 0x4000 allocated by thread T3:",
      s.data());
}

TEST(Report, DescribeLocationFD) {
  ReportLocation loc;
  loc.type = ReportLocationFD;
  loc.tid = kMainTid;
  loc.fd = 7;

  InternalScopedString s;
  loc.fd_closed = false;
  EXPECT_TRUE(DescribeLocation(&loc, s));
  EXPECT_STREQ("  Location is file descriptor 7 created by main thread at:",
               s.data());

  s.clear();
  loc.fd_closed = true;
  EXPECT_TRUE(DescribeLocation(&loc, s));
  EXPECT_STREQ("  Location is file descriptor 7 destroyed by main thread at:",
               s.data());
}

TEST(Report, DescribeMutex) {
  ReportMutex rm;
  rm.id = 5;
  rm.addr = 0xdead;

  InternalScopedString s;
  DescribeMutex(&rm, s);
  EXPECT_STREQ("  Mutex M5 (0xdead) created at:", s.data());
}

TEST(Report, DescribeThread) {
  ReportThread rt = {};
  rt.id = 1;
  rt.os_id = 42;
  rt.running = true;
  rt.thread_type = ThreadType::Regular;
  rt.name = nullptr;
  rt.parent_tid = kMainTid;
  rt.stack = nullptr;

  InternalScopedString s;
  // A regular thread is followed by its creation stack.
  EXPECT_TRUE(DescribeThread(&rt, s));
  EXPECT_STREQ("  Thread T1 (tid=42, running) created by main thread",
               s.data());

  // A name is included and " at:" is appended when a stack is present.
  s.clear();
  char name[] = "worker";
  rt.name = name;
  rt.running = false;
  ReportStack stack;
  rt.stack = &stack;
  EXPECT_TRUE(DescribeThread(&rt, s));
  EXPECT_STREQ(
      "  Thread T1 'worker' (tid=42, finished) created by main thread at:",
      s.data());

  // GCD worker threads have no following stack.
  s.clear();
  rt.name = nullptr;
  rt.running = true;
  rt.stack = nullptr;
  rt.thread_type = ThreadType::Worker;
  EXPECT_FALSE(DescribeThread(&rt, s));
  EXPECT_STREQ("  Thread T1 (tid=42, running) is a GCD worker thread",
               s.data());
}

}  // namespace __tsan
