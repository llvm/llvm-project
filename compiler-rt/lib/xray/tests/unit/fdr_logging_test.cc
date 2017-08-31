//===-- fdr_logging_test.cc -----------------------------------------------===//
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
#include "xray_fdr_logging.h"
#include "gtest/gtest.h"

#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <system_error>
#include <unistd.h>

#include "xray/xray_records.h"

namespace __xray {
namespace {

constexpr auto kBufferSize = 16384;
constexpr auto kBufferMax = 10;

struct ScopedFileCloserAndDeleter {
  explicit ScopedFileCloserAndDeleter(int Fd, const char *Filename)
      : Fd(Fd), Filename(Filename) {}

  ~ScopedFileCloserAndDeleter() {
    if (Fd) {
      close(Fd);
      unlink(Filename);
    }
  }

  int Fd;
  const char *Filename;
};

TEST(FDRLoggingTest, Simple) {
  FDRLoggingOptions Options;
  Options.ReportErrors = true;
  char TmpFilename[] = "fdr-logging-test.XXXXXX";
  Options.Fd = mkstemp(TmpFilename);
  ASSERT_NE(Options.Fd, -1);
  ASSERT_EQ(fdrLoggingInit(kBufferSize, kBufferMax, &Options,
                            sizeof(FDRLoggingOptions)),
            XRayLogInitStatus::XRAY_LOG_INITIALIZED);
  fdrLoggingHandleArg0(1, XRayEntryType::ENTRY);
  fdrLoggingHandleArg0(1, XRayEntryType::EXIT);
  ASSERT_EQ(fdrLoggingFinalize(), XRayLogInitStatus::XRAY_LOG_FINALIZED);
  ASSERT_EQ(fdrLoggingFlush(), XRayLogFlushStatus::XRAY_LOG_FLUSHED);

  // To do this properly, we have to close the file descriptor then re-open the
  // file for reading this time.
  ASSERT_EQ(close(Options.Fd), 0);
  int Fd = open(TmpFilename, O_RDONLY);
  ASSERT_NE(-1, Fd);
  ScopedFileCloserAndDeleter Guard(Fd, TmpFilename);
  auto Size = lseek(Fd, 0, SEEK_END);
  ASSERT_NE(Size, 0);
  // Map the file contents.
  const char *Contents = static_cast<const char *>(
      mmap(NULL, Size, PROT_READ, MAP_PRIVATE, Fd, 0));
  ASSERT_NE(Contents, nullptr);

  XRayFileHeader H;
  memcpy(&H, Contents, sizeof(XRayFileHeader));
  ASSERT_EQ(H.Version, 1);
  ASSERT_EQ(H.Type, FileTypes::FDR_LOG);

  // We require one buffer at least to have the "start of buffer" metadata
  // record.
  MetadataRecord MDR;
  memcpy(&MDR, Contents + sizeof(XRayFileHeader), sizeof(MetadataRecord));
  ASSERT_EQ(MDR.RecordKind, uint8_t(MetadataRecord::RecordKinds::NewBuffer));
}

TEST(FDRLoggingTest, Multiple) {
  FDRLoggingOptions Options;
  char TmpFilename[] = "fdr-logging-test.XXXXXX";
  Options.Fd = mkstemp(TmpFilename);
  ASSERT_NE(Options.Fd, -1);
  ASSERT_EQ(fdrLoggingInit(kBufferSize, kBufferMax, &Options,
                            sizeof(FDRLoggingOptions)),
            XRayLogInitStatus::XRAY_LOG_INITIALIZED);
  for (uint64_t I = 0; I < 100; ++I) {
    fdrLoggingHandleArg0(1, XRayEntryType::ENTRY);
    fdrLoggingHandleArg0(1, XRayEntryType::EXIT);
  }
  ASSERT_EQ(fdrLoggingFinalize(), XRayLogInitStatus::XRAY_LOG_FINALIZED);
  ASSERT_EQ(fdrLoggingFlush(), XRayLogFlushStatus::XRAY_LOG_FLUSHED);

  // To do this properly, we have to close the file descriptor then re-open the
  // file for reading this time.
  ASSERT_EQ(close(Options.Fd), 0);
  int Fd = open(TmpFilename, O_RDONLY);
  ASSERT_NE(-1, Fd);
  ScopedFileCloserAndDeleter Guard(Fd, TmpFilename);
  auto Size = lseek(Fd, 0, SEEK_END);
  ASSERT_NE(Size, 0);
  // Map the file contents.
  const char *Contents = static_cast<const char *>(
      mmap(NULL, Size, PROT_READ, MAP_PRIVATE, Fd, 0));
  ASSERT_NE(Contents, nullptr);

  XRayFileHeader H;
  memcpy(&H, Contents, sizeof(XRayFileHeader));
  ASSERT_EQ(H.Version, 1);
  ASSERT_EQ(H.Type, FileTypes::FDR_LOG);

  MetadataRecord MDR0;
  memcpy(&MDR0, Contents + sizeof(XRayFileHeader), sizeof(MetadataRecord));
  ASSERT_EQ(MDR0.RecordKind, uint8_t(MetadataRecord::RecordKinds::NewBuffer));
}

} // namespace
} // namespace __xray
