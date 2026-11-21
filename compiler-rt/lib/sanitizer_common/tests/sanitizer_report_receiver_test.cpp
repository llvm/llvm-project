//===-- sanitizer_report_receiver_test.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for sanitizer_report_receiver.cpp
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_report_receiver.h"

#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace __sanitizer {

namespace {

// Records every event as a string.
class RecordingReceiver : public ReportReceiver {
 public:
  void OnStart(const char* sanitizer, const char* type) override {
    events.push_back(std::string("start ") + sanitizer + " " + type);
  }
  void OnTitle(const char* title) override {
    events.push_back("title " + std::string(title));
  }
  void OnAddress(uptr addr, uptr size, ReportAddressKind kind,
                 const char* desc) override {
    events.push_back("address " + std::to_string(addr) + " " +
                     std::to_string(size) + " " + std::to_string(kind) + " " +
                     std::string(desc));
  }
  void OnStack(const uptr* frames, uptr num_frames, ReportStackKind kind,
               u64 tid, const char* desc) override {
    std::string e = "stack";
    for (uptr i = 0; i < num_frames; i++) e += " " + std::to_string(frames[i]);
    e += " " + std::to_string(kind) + " t" + std::to_string(tid) + " " +
         std::string(desc);
    events.push_back(e);
  }
  void OnFinish() override { events.push_back("finish"); }

  std::vector<std::string> events;
};

// Receivers cannot be unregistered, so share one across all tests.
RecordingReceiver* GetReceiver() {
  static RecordingReceiver* receiver = [] {
    auto* r = new RecordingReceiver;
    RegisterReportReceiver(r);
    return r;
  }();
  return receiver;
}

}  // namespace

TEST(ReportReceiver, EventsInOrder) {
  RecordingReceiver* r = GetReceiver();
  r->events.clear();
  {
    ScopedSanitizerReport srep("TestSanitizer", "test-error",
                               /*title_with_pid=*/false);
    srep.PrintTitleF("", "title %d", 1);
    srep.NotifyAddress(0x1000, 16, kReportAddressFault, "fault");
    const uptr frames[] = {1, 2, 3};
    srep.NotifyStack(frames, 3, kReportStackAllocation, "alloc", 77);
    srep.NotifyStack(frames, 1, kReportStackOther, "cur", 5);
  }
  ASSERT_EQ(r->events.size(), 6u);
  EXPECT_EQ(r->events[0], "start TestSanitizer test-error");
  EXPECT_EQ(r->events[1], "title title 1");
  EXPECT_EQ(r->events[2], "address 4096 16 1 fault");
  EXPECT_EQ(r->events[3], "stack 1 2 3 1 t77 alloc");
  EXPECT_EQ(r->events[4], "stack 1 0 t5 cur");
  EXPECT_EQ(r->events[5], "finish");
}

TEST(ReportReceiver, EmptyAddressAndStackAreIgnored) {
  RecordingReceiver* r = GetReceiver();
  r->events.clear();
  {
    ScopedSanitizerReport srep("TestSanitizer", "test-error",
                               /*title_with_pid=*/false);
    srep.NotifyAddress(0, 16, kReportAddressFault, "fault");
    srep.NotifyStack(nullptr, 0, kReportStackOther, nullptr, 0);
  }
  ASSERT_EQ(r->events.size(), 2u);
  EXPECT_EQ(r->events[0], "start TestSanitizer test-error");
  EXPECT_EQ(r->events[1], "finish");
}

TEST(ReportReceiver, PrintTitleFNotifiesFormattedTitle) {
  RecordingReceiver* r = GetReceiver();
  r->events.clear();
  {
    ScopedSanitizerReport srep("TestSanitizer", "test-error",
                               /*title_with_pid=*/false);
    srep.PrintTitleF("", "ERROR: TestSanitizer: %s on %p", "bad", (void*)0x10);
  }
  ASSERT_EQ(r->events.size(), 3u);
  EXPECT_EQ(r->events[1].substr(0, 31), "title ERROR: TestSanitizer: bad");
}

}  // namespace __sanitizer
