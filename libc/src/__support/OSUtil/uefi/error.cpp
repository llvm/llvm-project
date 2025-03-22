//===---------- UEFI implementation of error utils ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------===//

#include "error.h"
#include "errno.h"
#include "limits.h"
#include "src/__support/macros/config.h"

#define ERROR_BIT (sizeof(size_t) * CHAR_BIT)

namespace LIBC_NAMESPACE_DECL {

static constexpr struct errno_efi_status_entry {
  EFI_STATUS status;
  int errno_value;
} uefi_status_errno_map[] = {
    {EFI_SUCCESS, 0},
    {ERROR_BIT | EFI_LOAD_ERROR, EINVAL},
    {ERROR_BIT | EFI_INVALID_PARAMETER, EINVAL},
    {ERROR_BIT | EFI_BAD_BUFFER_SIZE, EINVAL},
    {ERROR_BIT | EFI_NOT_READY, EBUSY},
    {ERROR_BIT | EFI_DEVICE_ERROR, EIO},
    {ERROR_BIT | EFI_WRITE_PROTECTED, EPERM},
    {ERROR_BIT | EFI_OUT_OF_RESOURCES, ENOMEM},
    {ERROR_BIT | EFI_VOLUME_CORRUPTED, EROFS},
    {ERROR_BIT | EFI_VOLUME_FULL, ENOSPC},
    {ERROR_BIT | EFI_NO_MEDIA, ENODEV},
    {ERROR_BIT | EFI_MEDIA_CHANGED, ENXIO},
    {ERROR_BIT | EFI_NOT_FOUND, ENOENT},
    {ERROR_BIT | EFI_ACCESS_DENIED, EACCES},
    {ERROR_BIT | EFI_NO_RESPONSE, EBUSY},
    {ERROR_BIT | EFI_NO_MAPPING, ENODEV},
    {ERROR_BIT | EFI_TIMEOUT, EBUSY},
    {ERROR_BIT | EFI_NOT_STARTED, EAGAIN},
    {ERROR_BIT | EFI_ALREADY_STARTED, EINVAL},
    {ERROR_BIT | EFI_ABORTED, EFAULT},
    {ERROR_BIT | EFI_ICMP_ERROR, EIO},
    {ERROR_BIT | EFI_TFTP_ERROR, EIO},
    {ERROR_BIT | EFI_PROTOCOL_ERROR, EINVAL},
    {ERROR_BIT | EFI_INCOMPATIBLE_VERSION, EINVAL},
    {ERROR_BIT | EFI_SECURITY_VIOLATION, EPERM},
    {ERROR_BIT | EFI_CRC_ERROR, EINVAL},
    {ERROR_BIT | EFI_END_OF_MEDIA, EPIPE},
    {ERROR_BIT | EFI_END_OF_FILE, EPIPE},
    {ERROR_BIT | EFI_INVALID_LANGUAGE, EINVAL},
    {ERROR_BIT | EFI_COMPROMISED_DATA, EINVAL},
    {ERROR_BIT | EFI_IP_ADDRESS_CONFLICT, EINVAL},
    {ERROR_BIT | EFI_HTTP_ERROR, EIO},
    {EFI_WARN_UNKNOWN_GLYPH, EINVAL},
    {EFI_WARN_DELETE_FAILURE, EROFS},
    {EFI_WARN_WRITE_FAILURE, EROFS},
    {EFI_WARN_BUFFER_TOO_SMALL, E2BIG},
    {EFI_WARN_STALE_DATA, EINVAL},
    {EFI_WARN_FILE_SYSTEM, EROFS},
    {EFI_WARN_RESET_REQUIRED, EINTR},
};

static constexpr size_t uefi_status_errno_map_length =
    sizeof(uefi_status_errno_map) / sizeof(uefi_status_errno_map[0]);

int uefi_status_to_errno(EFI_STATUS status) {
  for (size_t i = 0; i < uefi_status_errno_map_length; i++) {
    const struct errno_efi_status_entry *entry = &uefi_status_errno_map[i];
    if (entry->status == status)
      return entry->errno_value;
  }

  // Unknown type
  __builtin_unreachable();
}

EFI_STATUS errno_to_uefi_status(int errno_value) {
  for (size_t i = 0; i < uefi_status_errno_map_length; i++) {
    const struct errno_efi_status_entry *entry = &uefi_status_errno_map[i];
    if (entry->errno_value == errno_value)
      return entry->status;
  }

  // Unknown type
  __builtin_unreachable();
}

} // namespace LIBC_NAMESPACE_DECL
