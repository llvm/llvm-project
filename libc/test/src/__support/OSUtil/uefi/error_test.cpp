//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for UEFI status macros and errno conversion utilities.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "include/llvm-libc-types/EFI_STATUS.h"
#include "src/__support/CPP/span.h"
#include "src/__support/OSUtil/uefi/error.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcUefiErrorTest, StatusBitMacros) {
  // Every entry after the leading EFI_SUCCESS one maps a failure.
  const LIBC_NAMESPACE::cpp::span<const LIBC_NAMESPACE::UefiStatusErrnoEntry>
      failures(LIBC_NAMESPACE::UEFI_STATUS_ERRNO_MAP.data() + 1,
               LIBC_NAMESPACE::UEFI_STATUS_ERRNO_MAP.size() - 1);

  EXPECT_FALSE(EFI_ERROR(EFI_SUCCESS));

  for (const LIBC_NAMESPACE::UefiStatusErrnoEntry &entry : failures) {
    EXPECT_NE(entry.status, static_cast<EFI_STATUS>(EFI_SUCCESS));
    EXPECT_NE(entry.errno_value, 0);
  }

  EXPECT_TRUE(EFI_ERROR(EFI_LOAD_ERROR));
  EXPECT_TRUE(EFI_ERROR(EFI_INVALID_PARAMETER));
  EXPECT_TRUE(EFI_ERROR(EFI_UNSUPPORTED));
  EXPECT_TRUE(EFI_ERROR(EFI_BUFFER_TOO_SMALL));
  EXPECT_TRUE(EFI_ERROR(EFI_OUT_OF_RESOURCES));
  EXPECT_TRUE(EFI_ERROR(EFI_NOT_FOUND));
  EXPECT_TRUE(EFI_ERROR(EFI_ABORTED));
  EXPECT_TRUE(EFI_ERROR(EFI_HTTP_ERROR));

  EXPECT_FALSE(EFI_ERROR(EFI_WARN_UNKNOWN_GLYPH));
  EXPECT_FALSE(EFI_ERROR(EFI_WARN_DELETE_FAILURE));
  EXPECT_FALSE(EFI_ERROR(EFI_WARN_WRITE_FAILURE));
  EXPECT_FALSE(EFI_ERROR(EFI_WARN_BUFFER_TOO_SMALL));
  EXPECT_FALSE(EFI_ERROR(EFI_WARN_STALE_DATA));
  EXPECT_FALSE(EFI_ERROR(EFI_WARN_FILE_SYSTEM));
  EXPECT_FALSE(EFI_ERROR(EFI_WARN_RESET_REQUIRED));
}

TEST(LlvmLibcUefiErrorTest, StatusToErrno) {
  using LIBC_NAMESPACE::uefi_status_to_errno;

  EXPECT_EQ(uefi_status_to_errno(EFI_SUCCESS), 0);
  EXPECT_EQ(uefi_status_to_errno(EFI_INVALID_PARAMETER), EINVAL);
  EXPECT_EQ(uefi_status_to_errno(EFI_UNSUPPORTED), EINVAL);
  EXPECT_EQ(uefi_status_to_errno(EFI_BUFFER_TOO_SMALL), ERANGE);
  EXPECT_EQ(uefi_status_to_errno(EFI_OUT_OF_RESOURCES), ENOMEM);
  EXPECT_EQ(uefi_status_to_errno(EFI_NOT_FOUND), ENOENT);
  EXPECT_EQ(uefi_status_to_errno(EFI_WARN_BUFFER_TOO_SMALL), E2BIG);

  // Unknown error status maps to EIO; unknown warning status maps to 0.
  EXPECT_EQ(uefi_status_to_errno(EFIERR(999)), EIO);
  EXPECT_EQ(uefi_status_to_errno(static_cast<EFI_STATUS>(999)), 0);
}

TEST(LlvmLibcUefiErrorTest, ErrnoToStatus) {
  using LIBC_NAMESPACE::errno_to_uefi_status;

  EXPECT_EQ(errno_to_uefi_status(0), static_cast<EFI_STATUS>(EFI_SUCCESS));
  EXPECT_EQ(errno_to_uefi_status(EINVAL), EFI_INVALID_PARAMETER);
  EXPECT_EQ(errno_to_uefi_status(ERANGE), EFI_BUFFER_TOO_SMALL);
  EXPECT_EQ(errno_to_uefi_status(ENOMEM), EFI_OUT_OF_RESOURCES);
  EXPECT_EQ(errno_to_uefi_status(ENOENT), EFI_NOT_FOUND);
  EXPECT_EQ(errno_to_uefi_status(-1), EFI_INVALID_PARAMETER);
}
