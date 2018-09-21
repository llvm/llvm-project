//===-- fdr_log_writer_test.cc --------------------------------------------===//
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
#include <time.h>

#include "xray/xray_records.h"
#include "xray_fdr_log_writer.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/XRay/Trace.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace __xray {
namespace {

static constexpr size_t kSize = 4096;

using ::llvm::HasValue;
using ::testing::Eq;
using ::testing::SizeIs;

// Exercise the common code path where we initialize a buffer and are able to
// write some records successfully.
TEST(FdrLogWriterTest, WriteSomeRecords) {
  bool Success = false;
  BufferQueue Buffers(kSize, 1, Success);
  BufferQueue::Buffer B;
  ASSERT_EQ(Buffers.getBuffer(B), BufferQueue::ErrorCode::Ok);

  FDRLogWriter Writer(B);
  MetadataRecord Preamble[] = {
      createMetadataRecord<MetadataRecord::RecordKinds::NewBuffer>(int32_t{1}),
      createMetadataRecord<MetadataRecord::RecordKinds::WalltimeMarker>(
          int64_t{1}, int32_t{2}),
      createMetadataRecord<MetadataRecord::RecordKinds::Pid>(int32_t{1}),
  };
  ASSERT_THAT(Writer.writeMetadataRecords(Preamble),
              Eq(sizeof(MetadataRecord) * 3));
  ASSERT_TRUE(Writer.writeMetadata<MetadataRecord::RecordKinds::NewCPUId>(1));
  ASSERT_TRUE(
      Writer.writeFunction(FDRLogWriter::FunctionRecordKind::Enter, 1, 1));
  ASSERT_TRUE(
      Writer.writeFunction(FDRLogWriter::FunctionRecordKind::Exit, 1, 1));
  ASSERT_EQ(Buffers.releaseBuffer(B), BufferQueue::ErrorCode::Ok);
  ASSERT_EQ(B.Data, nullptr);
  ASSERT_EQ(Buffers.finalize(), BufferQueue::ErrorCode::Ok);

  // We then need to go through each element of the Buffers, and re-create a
  // flat buffer that we would see if they were laid out in a file. This also
  // means we need to write out the header manually.
  // TODO: Isolate the file header writing.
  std::string Serialized;
  std::aligned_storage<sizeof(XRayFileHeader), alignof(XRayFileHeader)>::type
      HeaderStorage;
  auto *Header = reinterpret_cast<XRayFileHeader *>(&HeaderStorage);
  new (Header) XRayFileHeader();
  Header->Version = 3;
  Header->Type = FileTypes::FDR_LOG;
  Header->CycleFrequency = 3e9;
  Header->ConstantTSC = 1;
  Header->NonstopTSC = 1;
  Serialized.append(reinterpret_cast<const char *>(&HeaderStorage),
                    sizeof(XRayFileHeader));
  size_t BufferCount = 0;
  Buffers.apply([&](const BufferQueue::Buffer &B) {
    ++BufferCount;
    auto Size = atomic_load_relaxed(&B.Extents);
    auto Extents =
        createMetadataRecord<MetadataRecord::RecordKinds::BufferExtents>(Size);
    Serialized.append(reinterpret_cast<const char *>(&Extents),
                      sizeof(Extents));
    Serialized.append(reinterpret_cast<const char *>(B.Data), Size);
  });
  ASSERT_EQ(BufferCount, 1u);

  llvm::DataExtractor DE(Serialized, true, 8);
  auto TraceOrErr = llvm::xray::loadTrace(DE);
  EXPECT_THAT_EXPECTED(TraceOrErr, HasValue(SizeIs(2)));
}

} // namespace
} // namespace __xray
