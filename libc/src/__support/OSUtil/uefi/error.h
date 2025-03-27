//===----------- UEFI implementation of error utils --------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_UEFI_ERROR_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_UEFI_ERROR_H

#include <errno.h>
#include <limits.h>
#include "include/llvm-libc-types/EFI_STATUS.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

#define EFI_ERROR_MAX_BIT (1 << (sizeof(EFI_STATUS) * sizeof(char) - 1))
#define EFI_ENCODE_ERROR(value)                                                \
  (EFI_ERROR_MAX_BIT | (EFI_ERROR_MAX_BIT >> 2) | (value))
#define EFI_ENCODE_WARNING(value) ((EFI_ERROR_MAX_BIT >> 2) | (value))

static constexpr struct errno_efi_status_entry {
  EFI_STATUS status;
  int errno_value;
} UEFI_STATUS_ERRNO_MAP[] = {
    {EFI_SUCCESS, 0},
    {EFI_ENCODE_ERROR(EFI_LOAD_ERROR), EINVAL},
    {EFI_ENCODE_ERROR(EFI_INVALID_PARAMETER), EINVAL},
    {EFI_ENCODE_ERROR(EFI_BAD_BUFFER_SIZE), EINVAL},
    {EFI_ENCODE_ERROR(EFI_NOT_READY), EBUSY},
    {EFI_ENCODE_ERROR(EFI_DEVICE_ERROR), EIO},
    {EFI_ENCODE_ERROR(EFI_WRITE_PROTECTED), EPERM},
    {EFI_ENCODE_ERROR(EFI_OUT_OF_RESOURCES), ENOMEM},
    {EFI_ENCODE_ERROR(EFI_VOLUME_CORRUPTED), EROFS},
    {EFI_ENCODE_ERROR(EFI_VOLUME_FULL), ENOSPC},
    {EFI_ENCODE_ERROR(EFI_NO_MEDIA), ENODEV},
    {EFI_ENCODE_ERROR(EFI_MEDIA_CHANGED), ENXIO},
    {EFI_ENCODE_ERROR(EFI_NOT_FOUND), ENOENT},
    {EFI_ENCODE_ERROR(EFI_ACCESS_DENIED), EACCES},
    {EFI_ENCODE_ERROR(EFI_NO_RESPONSE), EBUSY},
    {EFI_ENCODE_ERROR(EFI_NO_MAPPING), ENODEV},
    {EFI_ENCODE_ERROR(EFI_TIMEOUT), EBUSY},
    {EFI_ENCODE_ERROR(EFI_NOT_STARTED), EAGAIN},
    {EFI_ENCODE_ERROR(EFI_ALREADY_STARTED), EINVAL},
    {EFI_ENCODE_ERROR(EFI_ABORTED), EFAULT},
    {EFI_ENCODE_ERROR(EFI_ICMP_ERROR), EIO},
    {EFI_ENCODE_ERROR(EFI_TFTP_ERROR), EIO},
    {EFI_ENCODE_ERROR(EFI_PROTOCOL_ERROR), EINVAL},
    {EFI_ENCODE_ERROR(EFI_INCOMPATIBLE_VERSION), EINVAL},
    {EFI_ENCODE_ERROR(EFI_SECURITY_VIOLATION), EPERM},
    {EFI_ENCODE_ERROR(EFI_CRC_ERROR), EINVAL},
    {EFI_ENCODE_ERROR(EFI_END_OF_MEDIA), EPIPE},
    {EFI_ENCODE_ERROR(EFI_END_OF_FILE), EPIPE},
    {EFI_ENCODE_ERROR(EFI_INVALID_LANGUAGE), EINVAL},
    {EFI_ENCODE_ERROR(EFI_COMPROMISED_DATA), EINVAL},
    {EFI_ENCODE_ERROR(EFI_IP_ADDRESS_CONFLICT), EINVAL},
    {EFI_ENCODE_ERROR(EFI_HTTP_ERROR), EIO},
    {EFI_ENCODE_WARNING(EFI_WARN_UNKNOWN_GLYPH), EINVAL},
    {EFI_ENCODE_WARNING(EFI_WARN_DELETE_FAILURE), EROFS},
    {EFI_ENCODE_WARNING(EFI_WARN_WRITE_FAILURE), EROFS},
    {EFI_ENCODE_WARNING(EFI_WARN_BUFFER_TOO_SMALL), E2BIG},
    {EFI_ENCODE_WARNING(EFI_WARN_STALE_DATA), EINVAL},
    {EFI_ENCODE_WARNING(EFI_WARN_FILE_SYSTEM), EROFS},
    {EFI_ENCODE_WARNING(EFI_WARN_RESET_REQUIRED), EINTR},
};

static constexpr size_t UEFI_STATUS_ERRNO_MAP_LENGTH =
    sizeof(UEFI_STATUS_ERRNO_MAP) / sizeof(UEFI_STATUS_ERRNO_MAP[0]);

static inline int uefi_status_to_errno(EFI_STATUS status) {
  for (size_t i = 0; i < UEFI_STATUS_ERRNO_MAP_LENGTH; i++) {
    const struct errno_efi_status_entry *entry = &UEFI_STATUS_ERRNO_MAP[i];
    if (entry->status == status)
      return entry->errno_value;
  }

  // Unknown type
  __builtin_unreachable();
}

static inline EFI_STATUS errno_to_uefi_status(int errno_value) {
  for (size_t i = 0; i < UEFI_STATUS_ERRNO_MAP_LENGTH; i++) {
    const struct errno_efi_status_entry *entry = &UEFI_STATUS_ERRNO_MAP[i];
    if (entry->errno_value == errno_value)
      return entry->status;
  }

  // Unknown type
  __builtin_unreachable();
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_UEFI_ERROR_H
