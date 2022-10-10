//===-- Implementation of a class for mapping errors to strings -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/StringUtil/error_to_string.h"

#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/stringstream.h"
#include "src/__support/StringUtil/message_mapper.h"
#include "src/__support/integer_to_string.h"

#include <errno.h>
#include <stddef.h>

namespace __llvm_libc {
namespace internal {

constexpr size_t max_buff_size() {
  constexpr size_t unknown_str_len = sizeof("Unknown error");
  constexpr size_t max_num_len =
      __llvm_libc::IntegerToString::dec_bufsize<int>();
  // the buffer should be able to hold "Unknown error" + ' ' + num_str
  return (unknown_str_len + 1 + max_num_len) * sizeof(char);
}

// This is to hold error strings that have to be custom built. It may be
// rewritten on every call to strerror (or other error to string function).
constexpr size_t ERR_BUFFER_SIZE = max_buff_size();
thread_local char error_buffer[ERR_BUFFER_SIZE];

// Since the StringMappings array is a map from error numbers to their
// corresponding strings, we have to have an array large enough we can use the
// error numbers as indexes. Thankfully there are 132 errors in the above list
// (41 and 58 are skipped) and the highest number is 133. If other platforms use
// different error numbers, then this number may need to be adjusted.
// Also if negative numbers or particularly large numbers are used, then the
// array should be turned into a proper hashmap.
constexpr size_t ERR_ARRAY_SIZE = 134;

constexpr MsgMapping raw_err_array[] = {
    MsgMapping(0, "Success"),
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
    MsgMapping(ENOTBLK, "Block device required"),
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
    MsgMapping(EDOM, "Numerical argument out of domain"),
    MsgMapping(ERANGE, "Numerical result out of range"),
    MsgMapping(EDEADLK, "Resource deadlock avoided"),
    MsgMapping(ENAMETOOLONG, "File name too long"),
    MsgMapping(ENOLCK, "No locks available"),
    MsgMapping(ENOSYS, "Function not implemented"),
    MsgMapping(ENOTEMPTY, "Directory not empty"),
    MsgMapping(ELOOP, "Too many levels of symbolic links"),
    // No error for 41. Would be EWOULDBLOCK
    MsgMapping(ENOMSG, "No message of desired type"),
    MsgMapping(EIDRM, "Identifier removed"),
    MsgMapping(ECHRNG, "Channel number out of range"),
    MsgMapping(EL2NSYNC, "Level 2 not synchronized"),
    MsgMapping(EL3HLT, "Level 3 halted"),
    MsgMapping(EL3RST, "Level 3 reset"),
    MsgMapping(ELNRNG, "Link number out of range"),
    MsgMapping(EUNATCH, "Protocol driver not attached"),
    MsgMapping(ENOCSI, "No CSI structure available"),
    MsgMapping(EL2HLT, "Level 2 halted"),
    MsgMapping(EBADE, "Invalid exchange"),
    MsgMapping(EBADR, "Invalid request descriptor"),
    MsgMapping(EXFULL, "Exchange full"),
    MsgMapping(ENOANO, "No anode"),
    MsgMapping(EBADRQC, "Invalid request code"),
    MsgMapping(EBADSLT, "Invalid slot"),
    // No error for 58. Would be EDEADLOCK.
    MsgMapping(EBFONT, "Bad font file format"),
    MsgMapping(ENOSTR, "Device not a stream"),
    MsgMapping(ENODATA, "No data available"),
    MsgMapping(ETIME, "Timer expired"),
    MsgMapping(ENOSR, "Out of streams resources"),
    MsgMapping(ENONET, "Machine is not on the network"),
    MsgMapping(ENOPKG, "Package not installed"),
    MsgMapping(EREMOTE, "Object is remote"),
    MsgMapping(ENOLINK, "Link has been severed"),
    MsgMapping(EADV, "Advertise error"),
    MsgMapping(ESRMNT, "Srmount error"),
    MsgMapping(ECOMM, "Communication error on send"),
    MsgMapping(EPROTO, "Protocol error"),
    MsgMapping(EMULTIHOP, "Multihop attempted"),
    MsgMapping(EDOTDOT, "RFS specific error"),
    MsgMapping(EBADMSG, "Bad message"),
    MsgMapping(EOVERFLOW, "Value too large for defined data type"),
    MsgMapping(ENOTUNIQ, "Name not unique on network"),
    MsgMapping(EBADFD, "File descriptor in bad state"),
    MsgMapping(EREMCHG, "Remote address changed"),
    MsgMapping(ELIBACC, "Can not access a needed shared library"),
    MsgMapping(ELIBBAD, "Accessing a corrupted shared library"),
    MsgMapping(ELIBSCN, ".lib section in a.out corrupted"),
    MsgMapping(ELIBMAX, "Attempting to link in too many shared libraries"),
    MsgMapping(ELIBEXEC, "Cannot exec a shared library directly"),
    MsgMapping(EILSEQ, "Invalid or incomplete multibyte or wide character"),
    MsgMapping(ERESTART, "Interrupted system call should be restarted"),
    MsgMapping(ESTRPIPE, "Streams pipe error"),
    MsgMapping(EUSERS, "Too many users"),
    MsgMapping(ENOTSOCK, "Socket operation on non-socket"),
    MsgMapping(EDESTADDRREQ, "Destination address required"),
    MsgMapping(EMSGSIZE, "Message too long"),
    MsgMapping(EPROTOTYPE, "Protocol wrong type for socket"),
    MsgMapping(ENOPROTOOPT, "Protocol not available"),
    MsgMapping(EPROTONOSUPPORT, "Protocol not supported"),
    MsgMapping(ESOCKTNOSUPPORT, "Socket type not supported"),
    MsgMapping(ENOTSUP, "Operation not supported"),
    MsgMapping(EPFNOSUPPORT, "Protocol family not supported"),
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
    MsgMapping(ESHUTDOWN, "Cannot send after transport endpoint shutdown"),
    MsgMapping(ETOOMANYREFS, "Too many references: cannot splice"),
    MsgMapping(ETIMEDOUT, "Connection timed out"),
    MsgMapping(ECONNREFUSED, "Connection refused"),
    MsgMapping(EHOSTDOWN, "Host is down"),
    MsgMapping(EHOSTUNREACH, "No route to host"),
    MsgMapping(EALREADY, "Operation already in progress"),
    MsgMapping(EINPROGRESS, "Operation now in progress"),
    MsgMapping(ESTALE, "Stale file handle"),
    MsgMapping(EUCLEAN, "Structure needs cleaning"),
    MsgMapping(ENOTNAM, "Not a XENIX named type file"),
    MsgMapping(ENAVAIL, "No XENIX semaphores available"),
    MsgMapping(EISNAM, "Is a named type file"),
    MsgMapping(EREMOTEIO, "Remote I/O error"),
    MsgMapping(EDQUOT, "Disk quota exceeded"),
    MsgMapping(ENOMEDIUM, "No medium found"),
    MsgMapping(EMEDIUMTYPE, "Wrong medium type"),
    MsgMapping(ECANCELED, "Operation canceled"),
    MsgMapping(ENOKEY, "Required key not available"),
    MsgMapping(EKEYEXPIRED, "Key has expired"),
    MsgMapping(EKEYREVOKED, "Key has been revoked"),
    MsgMapping(EKEYREJECTED, "Key was rejected by service"),
    MsgMapping(EOWNERDEAD, "Owner died"),
    MsgMapping(ENOTRECOVERABLE, "State not recoverable"),
    MsgMapping(ERFKILL, "Operation not possible due to RF-kill"),
    MsgMapping(EHWPOISON, "Memory page has hardware error"),
};

constexpr size_t RAW_ARRAY_LEN = sizeof(raw_err_array) / sizeof(MsgMapping);
constexpr size_t TOTAL_STR_LEN = total_str_len(raw_err_array, RAW_ARRAY_LEN);

static constexpr MessageMapper<ERR_ARRAY_SIZE, TOTAL_STR_LEN>
    error_mapper(raw_err_array, RAW_ARRAY_LEN);

cpp::string_view build_error_string(int err_num, cpp::span<char> buffer) {
  // if the buffer can't hold "Unknown error" + ' ' + num_str, then just
  // return "Unknown error".
  if (buffer.size() <
      (sizeof("Unknown error") + 1 + IntegerToString::dec_bufsize<int>()))
    return const_cast<char *>("Unknown error");

  cpp::StringStream buffer_stream(
      {const_cast<char *>(buffer.data()), buffer.size()});
  buffer_stream << "Unknown error" << ' ' << err_num << '\0';
  return buffer_stream.str();
}

} // namespace internal

cpp::string_view get_error_string(int err_num) {
  return get_error_string(err_num,
                          {internal::error_buffer, internal::ERR_BUFFER_SIZE});
}

cpp::string_view get_error_string(int err_num, cpp::span<char> buffer) {
  auto opt_str = internal::error_mapper.get_str(err_num);
  if (opt_str)
    return *opt_str;
  else
    return internal::build_error_string(err_num, buffer);
}

} // namespace __llvm_libc
