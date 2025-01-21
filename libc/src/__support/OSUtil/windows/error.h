//===------------- Convert Win32 Error to POSIX ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_WINDOWS_WINERROR_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_WINDOWS_WINERROR_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winerror.h>

#include "hdr/errno_macros.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {

LIBC_INLINE constexpr int map_win_error_to_errno(DWORD error) {
  // Translation based on
  // https://github.com/llvm/llvm-project/blob/2b26ee6e790574e05c3c9a562bc37897daf0f384/libcxx/src/system_error.cpp

  // For the relation between errc and errno, see:
  // https://en.cppreference.com/w/cpp/error/errc
  switch (error) {
  case ERROR_ACCESS_DENIED:
    return EPERM;
  case ERROR_ALREADY_EXISTS:
    return EEXIST;
  case ERROR_BAD_NETPATH:
    return ENOENT;
  case ERROR_BAD_PATHNAME:
    return ENOENT;
  case ERROR_BAD_UNIT:
    return ENODEV;
  case ERROR_BROKEN_PIPE:
    return EPIPE;
  case ERROR_BUFFER_OVERFLOW:
    return ENAMETOOLONG;
  case ERROR_BUSY:
    return EBUSY;
  case ERROR_BUSY_DRIVE:
    return EBUSY;
  case ERROR_CANNOT_MAKE:
    return EPERM;
  case ERROR_CANTOPEN:
    return EIO;
  case ERROR_CANTREAD:
    return EIO;
  case ERROR_CANTWRITE:
    return EIO;
  case ERROR_CURRENT_DIRECTORY:
    return EPERM;
  case ERROR_DEV_NOT_EXIST:
    return ENODEV;
  case ERROR_DEVICE_IN_USE:
    return EBUSY;
  case ERROR_DIR_NOT_EMPTY:
    return ENOTEMPTY;
  case ERROR_DIRECTORY:
    return EINVAL;
  case ERROR_DISK_FULL:
    return ENOSPC;
  case ERROR_FILE_EXISTS:
    return EEXIST;
  case ERROR_FILE_NOT_FOUND:
    return ENOENT;
  case ERROR_HANDLE_DISK_FULL:
    return ENOSPC;
  case ERROR_INVALID_ACCESS:
    return EPERM;
  case ERROR_INVALID_DRIVE:
    return ENODEV;
  case ERROR_INVALID_FUNCTION:
    return ENOSYS;
  case ERROR_INVALID_HANDLE:
    return EINVAL;
  case ERROR_INVALID_NAME:
    return ENOENT;
  case ERROR_INVALID_PARAMETER:
    return EINVAL;
  case ERROR_LOCK_VIOLATION:
    return ENOLCK;
  case ERROR_LOCKED:
    return ENOLCK;
  case ERROR_NEGATIVE_SEEK:
    return EINVAL;
  case ERROR_NOACCESS:
    return EPERM;
  case ERROR_NOT_ENOUGH_MEMORY:
    return ENOMEM;
  case ERROR_NOT_READY:
    return EAGAIN;
  case ERROR_NOT_SAME_DEVICE:
    return EXDEV;
  case ERROR_NOT_SUPPORTED:
    return ENOTSUP;
  case ERROR_OPEN_FAILED:
    return EIO;
  case ERROR_OPEN_FILES:
    return EBUSY;
  case ERROR_OPERATION_ABORTED:
    return ECANCELED;
  case ERROR_OUTOFMEMORY:
    return ENOMEM;
  case ERROR_PATH_NOT_FOUND:
    return ENOENT;
  case ERROR_READ_FAULT:
    return EIO;
  case ERROR_REPARSE_TAG_INVALID:
    return EINVAL;
  case ERROR_RETRY:
    return EAGAIN;
  case ERROR_SEEK:
    return EIO;
  case ERROR_SHARING_VIOLATION:
    return EPERM;
  case ERROR_TOO_MANY_OPEN_FILES:
    return EMFILE;
  case ERROR_WRITE_FAULT:
    return EIO;
  case ERROR_WRITE_PROTECT:
    return EPERM;
  default:
    // For unrecognized errno, default to ENOSYS
    return ENOTSUP;
  }
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_WINDOWS_WINERROR_H
