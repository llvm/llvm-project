//===- unittests/Basic/AtomicLineLoggerTest.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/AtomicLineLogger.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Threading.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"
#include <thread>

using namespace clang;

TEST(AtomicLineLoggerTest, DisabledLoggerDoesNotCrash) {
  AtomicLineLogger Logger;
  Logger.log() << "this goes nowhere";

  // An empty logger should not crash.
  EXPECT_TRUE(true);
}

#ifndef _WIN32
TEST(AtomicLineLoggerTest, LogLineMoveConstructor) {
  llvm::unittest::TempDir Dir("atomic-logger-test", /*Unique=*/true);
  SmallString<128> LogPath(Dir.path());
  llvm::sys::path::append(LogPath, "test.log");

  {
    AtomicLineLogger Logger(LogPath);
    LogLine Original = Logger.log();
    LogLine Moved(std::move(Original));
    Moved << "after_move";
  }

  auto BufOrErr = llvm::MemoryBuffer::getFile(LogPath);
  ASSERT_TRUE(BufOrErr) << "Failed to read log file";
  StringRef Content = (*BufOrErr)->getBuffer();

  // Only one line should be written (from Moved, not from Original).
  EXPECT_EQ(Content.count('\n'), 1u);
  EXPECT_TRUE(Content.contains("after_move"));
}

TEST(AtomicLineLoggerTest, LogLinePIDTIDMsg) {
  llvm::unittest::TempDir Dir("atomic-logger-test", /*Unique=*/true);
  SmallString<128> LogPath(Dir.path());
  llvm::sys::path::append(LogPath, "test.log");

  {
    AtomicLineLogger Logger(LogPath);
    Logger.log() << "test_event: " << "some_file.pcm";
  }

  auto BufOrErr = llvm::MemoryBuffer::getFile(LogPath);
  ASSERT_TRUE(BufOrErr) << "Failed to read log file";
  StringRef Content = (*BufOrErr)->getBuffer();

  // Ends with message + newline.
  EXPECT_TRUE(Content.ends_with("test_event: some_file.pcm\n"));

  // Prefix has the form: "<timestamp> <pid> <tid>: "
  // Verify PID matches this process.
  std::string ExpectedPID = std::to_string(llvm::sys::Process::getProcessId());
  EXPECT_TRUE(Content.contains(ExpectedPID));

  // Verify TID is present.
  std::string ExpectedTID = std::to_string(llvm::get_threadid());
  EXPECT_TRUE(Content.contains(ExpectedTID));
}

TEST(AtomicLineLoggerTest, LogLineLogArray) {
  llvm::unittest::TempDir Dir("atomic-logger-test", /*Unique=*/true);
  SmallString<128> LogPath(Dir.path());
  llvm::sys::path::append(LogPath, "test.log");

  {
    AtomicLineLogger Logger(LogPath);
    SmallVector<std::string> Args = {"clang", "-cc1", "-x", "c"};
    Logger.log().logArray("cmd:", " ", ArrayRef<std::string>(Args));
  }

  auto BufOrErr = llvm::MemoryBuffer::getFile(LogPath);
  ASSERT_TRUE(BufOrErr) << "Failed to read log file";
  StringRef Content = (*BufOrErr)->getBuffer();
  EXPECT_TRUE(Content.contains("cmd: clang -cc1 -x c"));
}

TEST(AtomicLineLoggerTest, LogLineTimestamp) {
  llvm::unittest::TempDir Dir("atomic-logger-test", /*Unique=*/true);
  SmallString<128> LogPath(Dir.path());
  llvm::sys::path::append(LogPath, "test.log");

  // Test that the timestamp generated is always sandwiched between Before
  // and After to verify the correctness.
  auto Before = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count();

  {
    AtomicLineLogger Logger(LogPath);
    Logger.log() << "test_event";
  }

  auto After = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count();

  auto BufOrErr = llvm::MemoryBuffer::getFile(LogPath);
  ASSERT_TRUE(BufOrErr) << "Failed to read log file";
  StringRef Content = (*BufOrErr)->getBuffer();

  // Extract timestamp from "[<seconds>.<millis>]" prefix.
  ASSERT_TRUE(Content.starts_with("["));
  size_t CloseBracket = Content.find(']');
  ASSERT_NE(CloseBracket, StringRef::npos);
  StringRef TimestampStr = Content.slice(1, CloseBracket);

  // Parse "<seconds>.<millis>".
  auto [SecStr, MillisStr] = TimestampStr.split('.');
  uint64_t Seconds, Millis;
  ASSERT_FALSE(SecStr.getAsInteger(10, Seconds));
  ASSERT_FALSE(MillisStr.getAsInteger(10, Millis));

  uint64_t LoggedMillis = Seconds * 1000 + Millis;
  EXPECT_GE(LoggedMillis, (uint64_t)Before);
  EXPECT_LE(LoggedMillis, (uint64_t)After);
}

TEST(AtomicLineLoggerTest, SingleLineWrittenToFile) {
  // Create a temp directory and build a log file path inside it.
  llvm::unittest::TempDir Dir("atomic-logger-test", /*Unique=*/true);
  SmallString<128> LogPath(Dir.path());
  llvm::sys::path::append(LogPath, "test.log");

  {
    AtomicLineLogger Logger(LogPath);
    Logger.log() << "pcm_write: module.pcm";
  }
  // Logger destroyed here. Log file is written to disk.

  // Read the file back.
  auto BufOrErr = llvm::MemoryBuffer::getFile(LogPath);
  ASSERT_TRUE(BufOrErr) << "Failed to read log file";
  StringRef Content = (*BufOrErr)->getBuffer();

  // Verify the message is present and the line ends with a newline.
  EXPECT_TRUE(Content.contains("pcm_write: module.pcm"));
  EXPECT_TRUE(Content.ends_with("\n"));

  // Verify there is exactly one line.
  EXPECT_EQ(Content.count('\n'), 1u);
}

TEST(AtomicLineLoggerTest, ConcurrentWritesProduceCompleteLines) {
  llvm::unittest::TempDir Dir("atomic-logger-concurrent", /*Unique=*/true);
  SmallString<128> LogPath(Dir.path());
  llvm::sys::path::append(LogPath, "concurrent.log");

  // Testing concurrent writing of the log file.
  // Each logged message starts with the string `thread_`, and the message is
  // always 32 characters long.
  constexpr unsigned NumThreads = 8;
  constexpr unsigned LinesPerThread = 100;
  constexpr unsigned MessageLen = 32;

  {
    // Creating two loggers based on the same file to make sure
    // the write is still atomic.
    AtomicLineLogger LoggerOdd(LogPath);
    AtomicLineLogger LoggerEven(LogPath);

    std::vector<std::thread> Threads;
    for (unsigned I = 0; I < NumThreads; ++I) {
      Threads.emplace_back([&, I] {
        for (unsigned J = 0; J < LinesPerThread; ++J) {
          SmallString<64> Msg;
          llvm::raw_svector_ostream MsgOS(Msg);
          MsgOS << "thread_" << llvm::format("%02u", I) << "_line_"
                << llvm::format("%03u", J);
          // Pad to fixed width.
          while (Msg.size() < MessageLen)
            MsgOS << '_';
          if (I % 2)
            LoggerOdd.log() << Msg;
          else
            LoggerEven.log() << Msg;
        }
      });
    }
    for (auto &T : Threads)
      T.join();
  }
  // Loggers destroyed here. Log file is written to disk.

  auto BufOrErr = llvm::MemoryBuffer::getFile(LogPath);
  ASSERT_TRUE(BufOrErr) << "Failed to read log file";
  StringRef Content = (*BufOrErr)->getBuffer();

  SmallVector<StringRef> Lines;
  Content.split(Lines, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  EXPECT_EQ(Lines.size(), (size_t)(NumThreads * LinesPerThread));

  for (const auto &Line : Lines) {
    // For each line, we check the separator, message length, message start and
    // the prefix format to make sure no lines are interleved.

    // Split at ": " to separate prefix from message body.
    auto [Prefix, Body] = Line.split(": ");
    ASSERT_FALSE(Body.empty())
        << "Malformed line (no ': ' separator): " << Line.str();

    // Message body checks.
    EXPECT_EQ(Body.size(), (size_t)MessageLen)
        << "Wrong message length (interleaving?): " << Line.str();
    EXPECT_TRUE(Body.starts_with("thread_"))
        << "Corrupted message body: " << Line.str();

    // Prefix format: "[<seconds>.<millis>] <pid> <tid>"
    // Parse timestamp.
    EXPECT_TRUE(Prefix.starts_with("[")) << "Missing '[': " << Prefix.str();
    size_t CloseBracket = Prefix.find(']');
    ASSERT_NE(CloseBracket, StringRef::npos) << "Missing ']': " << Prefix.str();

    StringRef TimestampStr = Prefix.slice(1, CloseBracket);
    auto [SecStr, MillisStr] = TimestampStr.split('.');
    uint64_t Seconds, Millis;
    EXPECT_FALSE(SecStr.getAsInteger(10, Seconds))
        << "Bad seconds: " << SecStr.str();
    EXPECT_FALSE(MillisStr.getAsInteger(10, Millis))
        << "Bad millis: " << MillisStr.str();

    // Parse PID and TID from the rest: " <pid> <tid>"
    StringRef Rest = Prefix.substr(CloseBracket + 1).ltrim();
    auto [PidStr, TidStr] = Rest.split(' ');
    uint64_t Pid, Tid;
    EXPECT_FALSE(PidStr.getAsInteger(10, Pid)) << "Bad PID: " << PidStr.str();
    EXPECT_FALSE(TidStr.getAsInteger(10, Tid)) << "Bad TID: " << TidStr.str();

    // PID should match this process.
    EXPECT_EQ(Pid, (uint64_t)llvm::sys::Process::getProcessId());
  }
}
#endif
