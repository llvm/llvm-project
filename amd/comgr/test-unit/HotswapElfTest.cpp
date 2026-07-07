//===- HotswapElfTest.cpp - Unit tests for HotSwap ELF layer --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"
#include "comgr-test-elf-utils.h"
#include "gtest/gtest.h"

#include <cstring>
#include <limits>

using namespace COMGR::hotswap;

static std::vector<uint8_t> makeText(size_t Size = 16) {
  return std::vector<uint8_t>(Size, 0);
}

static unsigned readReservedSgprs(const std::vector<uint8_t> &Bytes,
                                  uint64_t KernelDescriptorOffset) {
  namespace hsa = llvm::amdhsa;

  uint32_t Rsrc1 = 0;
  std::memcpy(&Rsrc1,
              Bytes.data() + KernelDescriptorOffset +
                  offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              sizeof(Rsrc1));
  return (AMDHSA_BITS_GET(
              Rsrc1, hsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT) +
          1) *
         8;
}

// -- ElfView::create ----------------------------------------------------------

TEST(ElfView, RejectsTruncatedInput) {
  uint8_t Garbage[] = {0x7f, 'E', 'L', 'F', 0, 0, 0, 0};
  llvm::Expected<ElfView> ViewOrErr = ElfView::create(Garbage, sizeof(Garbage));
  EXPECT_FALSE((bool)ViewOrErr);
  llvm::consumeError(ViewOrErr.takeError());
}

TEST(ElfView, RejectsNonElfInput) {
  uint8_t NotElf[64] = {};
  llvm::Expected<ElfView> ViewOrErr = ElfView::create(NotElf, sizeof(NotElf));
  EXPECT_FALSE((bool)ViewOrErr);
  llvm::consumeError(ViewOrErr.takeError());
}

// -- ElfView::findKernelAtOffset ----------------------------------------------

TEST(ElfView, FindKernelAtOffsetResolvesNearestPrecedingForZeroSizeSymbol) {
  // AMDGPU kernel entry symbols frequently have st_size == 0 (the size lives on
  // the .kd object symbol), so an exact [st_value, st_value + st_size)
  // containment test never matches. The lookup must resolve via the
  // nearest-preceding STT_FUNC symbol instead.
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "zero_size_kernel";
  Opts.TextAddr = 0x1000;
  Opts.ZeroSizeKernelSym = true;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  // findKernelAtOffset takes a virtual address; at the entry and at an interior
  // offset the zero-size symbol still resolves.
  EXPECT_EQ(ViewOrErr->findKernelAtOffset(0x1000), "zero_size_kernel");
  EXPECT_EQ(ViewOrErr->findKernelAtOffset(0x1000 + 4), "zero_size_kernel");
  // An address before the symbol has no preceding function symbol to resolve.
  EXPECT_EQ(ViewOrErr->findKernelAtOffset(0x0FF0), "");
}

// -- ElfView::getKernelStaticLdsSize ------------------------------------------

TEST(ElfView, GetKernelStaticLdsSizeReturnsNulloptWhenKdMissing) {
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText());

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  EXPECT_EQ(ViewOrErr->getKernelStaticLdsSize("nonexistent_kernel"),
            std::nullopt);
}

TEST(ElfView, GetKernelStaticLdsSizeReadsLdsSizeFromKernelDescriptor) {
  static constexpr uint32_t TestLdsSize = 16384;

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.ElfType = llvm::ELF::ET_REL;
  Opts.KernelName = "test_kernel";
  Opts.TextAddr = 0;
  Opts.RodataAddr = 0;
  Opts.GroupSegmentFixedSize = TestLdsSize;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  std::optional<uint32_t> Lds =
      ViewOrErr->getKernelStaticLdsSize("test_kernel");
  ASSERT_TRUE(Lds.has_value());
  EXPECT_EQ(*Lds, TestLdsSize);
}

TEST(ElfView, KernelDescriptorsEnumeratesAndUpdatesEntryOffset) {
  namespace hsa = llvm::amdhsa;

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  std::vector<KernelDescriptorInfo> KDs = ViewOrErr->kernelDescriptors();
  ASSERT_EQ(KDs.size(), 1u);
  EXPECT_EQ(KDs[0].KernelName, "entry_kernel");
  EXPECT_EQ(KDs[0].VAddr, Obj.RodataAddr);
  EXPECT_EQ(KDs[0].EntryOffset, Obj.EntryOffset);
  EXPECT_EQ(ViewOrErr->getKernelDescriptorVAddr("entry_kernel"),
            Obj.RodataAddr);

  const int64_t NewOff = -128;
  ASSERT_TRUE(
      ViewOrErr->updateKernelDescriptorEntryOffset("entry_kernel", NewOff));
  int64_t ReadBack = 0;
  std::memcpy(
      &ReadBack,
      Obj.Bytes.data() + Obj.KernelDescriptorOffset +
          offsetof(hsa::kernel_descriptor_t, kernel_code_entry_byte_offset),
      sizeof(ReadBack));
  EXPECT_EQ(ReadBack, NewOff);

  ASSERT_TRUE(ViewOrErr->updateKernelDescriptorSgprCount("entry_kernel", 10));
  EXPECT_GE(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 10u);
}

TEST(ElfView, KernelDescriptorsSkipsKdWhenFileOffsetOverflows) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "overflow_kernel";
  Opts.RodataAddr = 0x1000;
  Opts.KernelDescriptorSymbolValue =
      std::numeric_limits<uint64_t>::max() - 0x20;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  EXPECT_TRUE(ViewOrErr->kernelDescriptors().empty());
  EXPECT_EQ(ViewOrErr->findKernelDescriptor("overflow_kernel"), nullptr);
}

TEST(ElfView, GrowWithTrampolinesShiftsAllocSectionSymbols) {
  static constexpr uint64_t GrowthBytes = 8;

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  Trampoline T;
  T.Bytes.assign(GrowthBytes, 0);
  std::vector<Trampoline> Trampolines;
  Trampolines.push_back(T);
  const uint8_t SNop[4] = {};
  std::unique_ptr<llvm::WritableMemoryBuffer> Out =
      ViewOrErr->growWithTrampolines(Trampolines, SNop);
  ASSERT_NE(Out, nullptr);

  uint8_t *OutData = reinterpret_cast<uint8_t *>(Out->getBufferStart());
  llvm::Expected<ElfView> OutView =
      ElfView::create(OutData, Out->getBufferSize());
  ASSERT_TRUE((bool)OutView) << llvm::toString(OutView.takeError());
  std::vector<KernelDescriptorInfo> KDs = OutView->kernelDescriptors();
  ASSERT_EQ(KDs.size(), 1u);
  EXPECT_EQ(KDs[0].VAddr, Obj.RodataAddr + GrowthBytes);
}

TEST(ElfView, UpdateKernelDescriptorSgprCountUpdatesMetadataAndDescriptor) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataSgprCount = 8;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  ASSERT_TRUE(ViewOrErr->updateKernelDescriptorSgprCount("entry_kernel", 10));
  std::optional<unsigned> MetadataSgprs =
      ViewOrErr->getKernelSgprCount("entry_kernel");
  ASSERT_TRUE(MetadataSgprs.has_value());
  EXPECT_EQ(*MetadataSgprs, 10u);
  EXPECT_GE(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 10u);
}

TEST(ElfView, UpdateKernelDescriptorSgprCountRejectsMissingMetadataCount) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataOmitSgprCount = true;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  EXPECT_FALSE(ViewOrErr->updateKernelDescriptorSgprCount("entry_kernel", 10));
  EXPECT_EQ(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 8u);
}

TEST(ElfView, UpdateKernelDescriptorSgprCountRejectsMissingMetadataKernel) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataKernelName = "other_kernel";
  Opts.MetadataSgprCount = 8;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  EXPECT_EQ(ViewOrErr->getKernelSgprCount("entry_kernel"), std::nullopt);
  EXPECT_FALSE(ViewOrErr->updateKernelDescriptorSgprCount("entry_kernel", 10));
  EXPECT_EQ(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 8u);
}

TEST(ElfView, UpdateKernelDescriptorSgprCountRejectsNonIntegerMetadataCount) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataSgprCountAsString = true;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  EXPECT_EQ(ViewOrErr->getKernelSgprCount("entry_kernel"), std::nullopt);
  EXPECT_FALSE(ViewOrErr->updateKernelDescriptorSgprCount("entry_kernel", 10));
  EXPECT_EQ(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 8u);
}

TEST(ElfView, UpdateKernelDescriptorSgprCountRejectsMetadataSizeChange) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataSgprCount = 9;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  EXPECT_FALSE(ViewOrErr->updateKernelDescriptorSgprCount("entry_kernel", 128));
  std::optional<unsigned> MetadataSgprs =
      ViewOrErr->getKernelSgprCount("entry_kernel");
  ASSERT_TRUE(MetadataSgprs.has_value());
  EXPECT_EQ(*MetadataSgprs, 9u);
  EXPECT_EQ(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 8u);
}

TEST(ElfView, UpdateKernelDescriptorSgprCountRejectsDescriptorLimitFirst) {
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.KernelName = "entry_kernel";
  Opts.MetadataSgprCount = 200;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(makeText(), Opts);

  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  EXPECT_FALSE(
      ViewOrErr->updateKernelDescriptorSgprCount("entry_kernel", 100000));
  std::optional<unsigned> MetadataSgprs =
      ViewOrErr->getKernelSgprCount("entry_kernel");
  ASSERT_TRUE(MetadataSgprs.has_value());
  EXPECT_EQ(*MetadataSgprs, 200u);
  EXPECT_EQ(readReservedSgprs(Obj.Bytes, Obj.KernelDescriptorOffset), 8u);
}
