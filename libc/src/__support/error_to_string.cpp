//===-- Implementation of a class for mapping errors to strings -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/error_to_string.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/stringstream.h"
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
constexpr size_t BUFFER_SIZE = max_buff_size();
thread_local char error_buffer[BUFFER_SIZE];

struct ErrMsgMapping {
  int err_num;
  cpp::string_view err_msg;

public:
  constexpr ErrMsgMapping(int num, const char *msg)
      : err_num(num), err_msg(msg) {
    ;
  }
};

constexpr ErrMsgMapping raw_err_array[] = {
    ErrMsgMapping(0, "Success"),
    ErrMsgMapping(EPERM, "Operation not permitted"),
    ErrMsgMapping(ENOENT, "No such file or directory"),
    ErrMsgMapping(ESRCH, "No such process"),
    ErrMsgMapping(EINTR, "Interrupted system call"),
    ErrMsgMapping(EIO, "Input/output error"),
    ErrMsgMapping(ENXIO, "No such device or address"),
    ErrMsgMapping(E2BIG, "Argument list too long"),
    ErrMsgMapping(ENOEXEC, "Exec format error"),
    ErrMsgMapping(EBADF, "Bad file descriptor"),
    ErrMsgMapping(ECHILD, "No child processes"),
    ErrMsgMapping(EAGAIN, "Resource temporarily unavailable"),
    ErrMsgMapping(ENOMEM, "Cannot allocate memory"),
    ErrMsgMapping(EACCES, "Permission denied"),
    ErrMsgMapping(EFAULT, "Bad address"),
    ErrMsgMapping(ENOTBLK, "Block device required"),
    ErrMsgMapping(EBUSY, "Device or resource busy"),
    ErrMsgMapping(EEXIST, "File exists"),
    ErrMsgMapping(EXDEV, "Invalid cross-device link"),
    ErrMsgMapping(ENODEV, "No such device"),
    ErrMsgMapping(ENOTDIR, "Not a directory"),
    ErrMsgMapping(EISDIR, "Is a directory"),
    ErrMsgMapping(EINVAL, "Invalid argument"),
    ErrMsgMapping(ENFILE, "Too many open files in system"),
    ErrMsgMapping(EMFILE, "Too many open files"),
    ErrMsgMapping(ENOTTY, "Inappropriate ioctl for device"),
    ErrMsgMapping(ETXTBSY, "Text file busy"),
    ErrMsgMapping(EFBIG, "File too large"),
    ErrMsgMapping(ENOSPC, "No space left on device"),
    ErrMsgMapping(ESPIPE, "Illegal seek"),
    ErrMsgMapping(EROFS, "Read-only file system"),
    ErrMsgMapping(EMLINK, "Too many links"),
    ErrMsgMapping(EPIPE, "Broken pipe"),
    ErrMsgMapping(EDOM, "Numerical argument out of domain"),
    ErrMsgMapping(ERANGE, "Numerical result out of range"),
    ErrMsgMapping(EDEADLK, "Resource deadlock avoided"),
    ErrMsgMapping(ENAMETOOLONG, "File name too long"),
    ErrMsgMapping(ENOLCK, "No locks available"),
    ErrMsgMapping(ENOSYS, "Function not implemented"),
    ErrMsgMapping(ENOTEMPTY, "Directory not empty"),
    ErrMsgMapping(ELOOP, "Too many levels of symbolic links"),
    // No error for 41. Would be EWOULDBLOCK
    ErrMsgMapping(ENOMSG, "No message of desired type"),
    ErrMsgMapping(EIDRM, "Identifier removed"),
    ErrMsgMapping(ECHRNG, "Channel number out of range"),
    ErrMsgMapping(EL2NSYNC, "Level 2 not synchronized"),
    ErrMsgMapping(EL3HLT, "Level 3 halted"),
    ErrMsgMapping(EL3RST, "Level 3 reset"),
    ErrMsgMapping(ELNRNG, "Link number out of range"),
    ErrMsgMapping(EUNATCH, "Protocol driver not attached"),
    ErrMsgMapping(ENOCSI, "No CSI structure available"),
    ErrMsgMapping(EL2HLT, "Level 2 halted"),
    ErrMsgMapping(EBADE, "Invalid exchange"),
    ErrMsgMapping(EBADR, "Invalid request descriptor"),
    ErrMsgMapping(EXFULL, "Exchange full"),
    ErrMsgMapping(ENOANO, "No anode"),
    ErrMsgMapping(EBADRQC, "Invalid request code"),
    ErrMsgMapping(EBADSLT, "Invalid slot"),
    // No error for 58. Would be EDEADLOCK.
    ErrMsgMapping(EBFONT, "Bad font file format"),
    ErrMsgMapping(ENOSTR, "Device not a stream"),
    ErrMsgMapping(ENODATA, "No data available"),
    ErrMsgMapping(ETIME, "Timer expired"),
    ErrMsgMapping(ENOSR, "Out of streams resources"),
    ErrMsgMapping(ENONET, "Machine is not on the network"),
    ErrMsgMapping(ENOPKG, "Package not installed"),
    ErrMsgMapping(EREMOTE, "Object is remote"),
    ErrMsgMapping(ENOLINK, "Link has been severed"),
    ErrMsgMapping(EADV, "Advertise error"),
    ErrMsgMapping(ESRMNT, "Srmount error"),
    ErrMsgMapping(ECOMM, "Communication error on send"),
    ErrMsgMapping(EPROTO, "Protocol error"),
    ErrMsgMapping(EMULTIHOP, "Multihop attempted"),
    ErrMsgMapping(EDOTDOT, "RFS specific error"),
    ErrMsgMapping(EBADMSG, "Bad message"),
    ErrMsgMapping(EOVERFLOW, "Value too large for defined data type"),
    ErrMsgMapping(ENOTUNIQ, "Name not unique on network"),
    ErrMsgMapping(EBADFD, "File descriptor in bad state"),
    ErrMsgMapping(EREMCHG, "Remote address changed"),
    ErrMsgMapping(ELIBACC, "Can not access a needed shared library"),
    ErrMsgMapping(ELIBBAD, "Accessing a corrupted shared library"),
    ErrMsgMapping(ELIBSCN, ".lib section in a.out corrupted"),
    ErrMsgMapping(ELIBMAX, "Attempting to link in too many shared libraries"),
    ErrMsgMapping(ELIBEXEC, "Cannot exec a shared library directly"),
    ErrMsgMapping(EILSEQ, "Invalid or incomplete multibyte or wide character"),
    ErrMsgMapping(ERESTART, "Interrupted system call should be restarted"),
    ErrMsgMapping(ESTRPIPE, "Streams pipe error"),
    ErrMsgMapping(EUSERS, "Too many users"),
    ErrMsgMapping(ENOTSOCK, "Socket operation on non-socket"),
    ErrMsgMapping(EDESTADDRREQ, "Destination address required"),
    ErrMsgMapping(EMSGSIZE, "Message too long"),
    ErrMsgMapping(EPROTOTYPE, "Protocol wrong type for socket"),
    ErrMsgMapping(ENOPROTOOPT, "Protocol not available"),
    ErrMsgMapping(EPROTONOSUPPORT, "Protocol not supported"),
    ErrMsgMapping(ESOCKTNOSUPPORT, "Socket type not supported"),
    ErrMsgMapping(ENOTSUP, "Operation not supported"),
    ErrMsgMapping(EPFNOSUPPORT, "Protocol family not supported"),
    ErrMsgMapping(EAFNOSUPPORT, "Address family not supported by protocol"),
    ErrMsgMapping(EADDRINUSE, "Address already in use"),
    ErrMsgMapping(EADDRNOTAVAIL, "Cannot assign requested address"),
    ErrMsgMapping(ENETDOWN, "Network is down"),
    ErrMsgMapping(ENETUNREACH, "Network is unreachable"),
    ErrMsgMapping(ENETRESET, "Network dropped connection on reset"),
    ErrMsgMapping(ECONNABORTED, "Software caused connection abort"),
    ErrMsgMapping(ECONNRESET, "Connection reset by peer"),
    ErrMsgMapping(ENOBUFS, "No buffer space available"),
    ErrMsgMapping(EISCONN, "Transport endpoint is already connected"),
    ErrMsgMapping(ENOTCONN, "Transport endpoint is not connected"),
    ErrMsgMapping(ESHUTDOWN, "Cannot send after transport endpoint shutdown"),
    ErrMsgMapping(ETOOMANYREFS, "Too many references: cannot splice"),
    ErrMsgMapping(ETIMEDOUT, "Connection timed out"),
    ErrMsgMapping(ECONNREFUSED, "Connection refused"),
    ErrMsgMapping(EHOSTDOWN, "Host is down"),
    ErrMsgMapping(EHOSTUNREACH, "No route to host"),
    ErrMsgMapping(EALREADY, "Operation already in progress"),
    ErrMsgMapping(EINPROGRESS, "Operation now in progress"),
    ErrMsgMapping(ESTALE, "Stale file handle"),
    ErrMsgMapping(EUCLEAN, "Structure needs cleaning"),
    ErrMsgMapping(ENOTNAM, "Not a XENIX named type file"),
    ErrMsgMapping(ENAVAIL, "No XENIX semaphores available"),
    ErrMsgMapping(EISNAM, "Is a named type file"),
    ErrMsgMapping(EREMOTEIO, "Remote I/O error"),
    ErrMsgMapping(EDQUOT, "Disk quota exceeded"),
    ErrMsgMapping(ENOMEDIUM, "No medium found"),
    ErrMsgMapping(EMEDIUMTYPE, "Wrong medium type"),
    ErrMsgMapping(ECANCELED, "Operation canceled"),
    ErrMsgMapping(ENOKEY, "Required key not available"),
    ErrMsgMapping(EKEYEXPIRED, "Key has expired"),
    ErrMsgMapping(EKEYREVOKED, "Key has been revoked"),
    ErrMsgMapping(EKEYREJECTED, "Key was rejected by service"),
    ErrMsgMapping(EOWNERDEAD, "Owner died"),
    ErrMsgMapping(ENOTRECOVERABLE, "State not recoverable"),
    ErrMsgMapping(ERFKILL, "Operation not possible due to RF-kill"),
    ErrMsgMapping(EHWPOISON, "Memory page has hardware error"),
};

constexpr size_t total_str_len(const ErrMsgMapping *array, size_t len) {
  size_t total = 0;
  for (size_t i = 0; i < len; ++i) {
    // add 1 for the null terminator.
    total += array[i].err_msg.size() + 1;
  }
  return total;
}

// Since the StringMappings array is a map from error numbers to their
// corresponding strings, we have to have an array large enough we can use the
// error numbers as indexes. Thankfully there are 132 errors in the above list
// (41 and 58 are skipped) and the highest number is 133. If other platforms use
// different error numbers, then this number may need to be adjusted.
// Also if negative numbers or particularly large numbers are used, then the
// array should be turned into a proper hashmap.
constexpr size_t ERR_ARRAY_SIZE = 134;

class ErrorMapper {

  // const char *StringMappings[ERR_ARRAY_SIZE] = {""};
  int err_offsets[ERR_ARRAY_SIZE] = {-1};
  char string_array[total_str_len(
      raw_err_array, sizeof(raw_err_array) / sizeof(ErrMsgMapping))] = {'\0'};

public:
  constexpr ErrorMapper() {
    cpp::string_view string_mappings[ERR_ARRAY_SIZE] = {""};
    for (size_t i = 0; i < (sizeof(raw_err_array) / sizeof(ErrMsgMapping)); ++i)
      string_mappings[raw_err_array[i].err_num] = raw_err_array[i].err_msg;

    size_t string_array_index = 0;
    for (size_t cur_err = 0; cur_err < ERR_ARRAY_SIZE; ++cur_err) {
      if (string_mappings[cur_err].size() != 0) {
        err_offsets[cur_err] = string_array_index;
        // No need to replace with proper strcpy, this is evaluated at compile
        // time.
        for (size_t i = 0; i < string_mappings[cur_err].size() + 1;
             ++i, ++string_array_index) {
          string_array[string_array_index] = string_mappings[cur_err][i];
        }
      } else {
        err_offsets[cur_err] = -1;
      }
    }
  }

  cpp::string_view get_str(int err_num) const {
    if (err_num >= 0 && static_cast<size_t>(err_num) < ERR_ARRAY_SIZE &&
        err_offsets[err_num] != -1) {
      return const_cast<char *>(string_array + err_offsets[err_num]);
    } else {
      // if the buffer can't hold "Unknown error" + ' ' + num_str, then just
      // return "Unknown error".
      if (BUFFER_SIZE <
          (sizeof("Unknown error") + 1 + IntegerToString::dec_bufsize<int>()))
        return const_cast<char *>("Unknown error");

      cpp::StringStream buffer_stream({error_buffer, BUFFER_SIZE});
      buffer_stream << "Unknown error" << ' ' << err_num << '\0';
      return buffer_stream.str();
    }
  }
};

static constexpr ErrorMapper error_mapper;

} // namespace internal

cpp::string_view get_error_string(int err_num) {
  return internal::error_mapper.get_str(err_num);
}
} // namespace __llvm_libc
