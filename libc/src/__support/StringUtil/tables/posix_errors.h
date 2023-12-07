//===-- Map of POSIX error numbers to strings -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_POSIX_ERRORS_H
#define LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_POSIX_ERRORS_H

#include "src/__support/StringUtil/message_mapper.h"

#include <errno.h> // For error macros

namespace __llvm_libc {

LIBC_INLINE_VAR constexpr MsgTable<76> POSIX_ERRORS = {
    MsgMapping(EPERM, "Operation not permitted"),
    MsgMapping(ENOENT, "No such file or directory"),
    MsgMapping(ESRCH, "No such process"),
    MsgMapping(EINTR, "Interrupted system call"),
    MsgMapping(EIO, "Input/output error"),
    MsgMapping(ENXIO, "No such device or address"),
    MsgMapping(E2BIG, "Argument list too long"),
    MsgMapping(ENOEXEC, "Exec format error"),
    MsgMapping(EBADF, "Bad file descriptor"),
    MsgMapping(ECHILD, "No child processes"),
    MsgMapping(EAGAIN, "Resource temporarily unavailable"),
    MsgMapping(ENOMEM, "Cannot allocate memory"),
    MsgMapping(EACCES, "Permission denied"),
    MsgMapping(EFAULT, "Bad address"),
    MsgMapping(EBUSY, "Device or resource busy"),
    MsgMapping(EEXIST, "File exists"),
    MsgMapping(EXDEV, "Invalid cross-device link"),
    MsgMapping(ENODEV, "No such device"),
    MsgMapping(ENOTDIR, "Not a directory"),
    MsgMapping(EISDIR, "Is a directory"),
    MsgMapping(EINVAL, "Invalid argument"),
    MsgMapping(ENFILE, "Too many open files in system"),
    MsgMapping(EMFILE, "Too many open files"),
    MsgMapping(ENOTTY, "Inappropriate ioctl for device"),
    MsgMapping(ETXTBSY, "Text file busy"),
    MsgMapping(EFBIG, "File too large"),
    MsgMapping(ENOSPC, "No space left on device"),
    MsgMapping(ESPIPE, "Illegal seek"),
    MsgMapping(EROFS, "Read-only file system"),
    MsgMapping(EMLINK, "Too many links"),
    MsgMapping(EPIPE, "Broken pipe"),
    MsgMapping(EDEADLK, "Resource deadlock avoided"),
    MsgMapping(ENAMETOOLONG, "File name too long"),
    MsgMapping(ENOLCK, "No locks available"),
    MsgMapping(ENOSYS, "Function not implemented"),
    MsgMapping(ENOTEMPTY, "Directory not empty"),
    MsgMapping(ELOOP, "Too many levels of symbolic links"),
    MsgMapping(ENOMSG, "No message of desired type"),
    MsgMapping(EIDRM, "Identifier removed"),
    MsgMapping(ENOSTR, "Device not a stream"),
    MsgMapping(ENODATA, "No data available"),
    MsgMapping(ETIME, "Timer expired"),
    MsgMapping(ENOSR, "Out of streams resources"),
    MsgMapping(ENOLINK, "Link has been severed"),
    MsgMapping(EPROTO, "Protocol error"),
    MsgMapping(EMULTIHOP, "Multihop attempted"),
    MsgMapping(EBADMSG, "Bad message"),
    MsgMapping(EOVERFLOW, "Value too large for defined data type"),
    MsgMapping(ENOTSOCK, "Socket operation on non-socket"),
    MsgMapping(EDESTADDRREQ, "Destination address required"),
    MsgMapping(EMSGSIZE, "Message too long"),
    MsgMapping(EPROTOTYPE, "Protocol wrong type for socket"),
    MsgMapping(ENOPROTOOPT, "Protocol not available"),
    MsgMapping(EPROTONOSUPPORT, "Protocol not supported"),
    MsgMapping(ENOTSUP, "Operation not supported"),
    MsgMapping(EAFNOSUPPORT, "Address family not supported by protocol"),
    MsgMapping(EADDRINUSE, "Address already in use"),
    MsgMapping(EADDRNOTAVAIL, "Cannot assign requested address"),
    MsgMapping(ENETDOWN, "Network is down"),
    MsgMapping(ENETUNREACH, "Network is unreachable"),
    MsgMapping(ENETRESET, "Network dropped connection on reset"),
    MsgMapping(ECONNABORTED, "Software caused connection abort"),
    MsgMapping(ECONNRESET, "Connection reset by peer"),
    MsgMapping(ENOBUFS, "No buffer space available"),
    MsgMapping(EISCONN, "Transport endpoint is already connected"),
    MsgMapping(ENOTCONN, "Transport endpoint is not connected"),
    MsgMapping(ETIMEDOUT, "Connection timed out"),
    MsgMapping(ECONNREFUSED, "Connection refused"),
    MsgMapping(EHOSTUNREACH, "No route to host"),
    MsgMapping(EALREADY, "Operation already in progress"),
    MsgMapping(EINPROGRESS, "Operation now in progress"),
    MsgMapping(ESTALE, "Stale file handle"),
    MsgMapping(EDQUOT, "Disk quota exceeded"),
    MsgMapping(ECANCELED, "Operation canceled"),
    MsgMapping(EOWNERDEAD, "Owner died"),
    MsgMapping(ENOTRECOVERABLE, "State not recoverable"),
};

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_STRING_UTIL_TABLES_POSIX_ERRORS_H
