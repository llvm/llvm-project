//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
/*
wine (wine_errc) error domain header.

Declares wine_errc (Wine's own UNIX errno values) and its error_domain
specialization. The singleton vtable is implemented in src/wine.cpp. Only
available on _WIN32/__CYGWIN__ targets.

The enumerator values match fast_io's
src/__wine_unix/include/__wine_unix/__wine_unix_errno.h (Linux-kernel style
numbering, referenced from the Linux kernel header). These are the values Wine
reports for host-side errors; they differ from the Windows API error codes and
are independent of the host libc's errno.
*/

namespace std {

enum class wine_errc : ::std::uint_least32_t {
  success = 0,                             // __WINE_UNIX_ERRNO_SUCCESS
  operation_not_permitted = 1,             // EPERM
  no_such_file_or_directory = 2,           // ENOENT
  no_such_process = 3,                     // ESRCH
  interrupted = 4,                         // EINTR
  io_error = 5,                            // EIO
  no_such_device_or_address = 6,           // ENXIO
  argument_list_too_long = 7,              // E2BIG
  executable_format_error = 8,             // ENOEXEC
  bad_file_descriptor = 9,                 // EBADF
  no_child_process = 10,                   // ECHILD
  resource_unavailable_try_again = 11,     // EAGAIN
  not_enough_memory = 12,                  // ENOMEM
  bad_address = 14,                        // EFAULT
  device_or_resource_busy = 16,            // EBUSY
  file_exists = 17,                        // EEXIST
  cross_device_link = 18,                  // EXDEV
  no_such_device = 19,                     // ENODEV
  not_a_directory = 20,                    // ENOTDIR
  is_a_directory = 21,                     // EISDIR
  invalid_argument = 22,                   // EINVAL
  too_many_files_open_in_system = 23,      // ENFILE
  too_many_files_open = 24,                // EMFILE
  inappropriate_io_control_operation = 25, // ENOTTY
  text_file_busy = 26,                     // ETXTBSY
  file_too_large = 27,                     // EFBIG
  no_space_on_device = 28,                 // ENOSPC
  invalid_seek = 29,                       // ESPIPE
  read_only_file_system = 30,              // EROFS
  too_many_links = 31,                     // EMLINK
  broken_pipe = 32,                        // EPIPE
  argument_out_of_domain = 33,             // EDOM
  result_out_of_range = 34,                // ERANGE
  resource_deadlock_would_occur = 35,      // EDEADLK
  filename_too_long = 36,                  // ENAMETOOLONG
  no_lock_available = 37,                  // ENOLCK
  function_not_supported = 38,             // ENOSYS
  directory_not_empty = 39,                // ENOTEMPTY
  too_many_symbolic_link_levels = 40,      // ELOOP
  would_block = 41,                        // EWOULDBLOCK
  no_message = 42,                         // ENOMSG
  identifier_removed = 43,                 // EIDRM
  no_link = 67,                            // ENOLINK
  protocol_error = 71,                     // EPROTO
  bad_message = 74,                        // EBADMSG
  value_too_large = 75,                    // EOVERFLOW
  illegal_byte_sequence = 84,              // EILSEQ
  not_a_socket = 88,                       // ENOTSOCK
  destination_address_required = 89,       // EDESTADDRREQ
  message_size = 90,                       // EMSGSIZE
  wrong_protocol_type = 91,                // EPROTOTYPE
  no_protocol_option = 92,                 // ENOPROTOOPT
  protocol_not_supported = 93,             // EPROTONOSUPPORT
  operation_not_supported = 95,            // EOPNOTSUPP
  address_family_not_supported = 97,       // EAFNOSUPPORT
  address_in_use = 98,                     // EADDRINUSE
  address_not_available = 99,              // EADDRNOTAVAIL
  network_down = 100,                      // ENETDOWN
  network_unreachable = 101,               // ENETUNREACH
  connection_aborted = 103,                // ECONNABORTED
  connection_reset = 104,                  // ECONNRESET
  no_buffer_space = 105,                   // ENOBUFS
  already_connected = 106,                 // EISCONN
  not_connected = 107,                     // ENOTCONN
  timed_out = 110,                         // ETIMEDOUT
  connection_refused = 111,                // ECONNREFUSED
  host_unreachable = 113,                  // EHOSTUNREACH
  operation_in_progress = 115,             // EINPROGRESS
  stale_file_handle = 116,                 // ESTALE
  operation_canceled = 125,                // ECANCELED
  owner_dead = 130,                        // EOWNERDEAD
  state_not_recoverable = 131,             // ENOTRECOVERABLE
};

inline constexpr bool operator==(::std::wine_errc __a,
                                 ::std::wine_errc __b) noexcept {
  return static_cast<::std::uint_least32_t>(__a) ==
         static_cast<::std::uint_least32_t>(__b);
}

template <> class error_domain<::std::wine_errc> {
public:
  using errc_type = ::std::wine_errc;
  static inline constexpr ::std::error_domain_singleton const *
  domain() noexcept {
    return ::std::error_domains::__cxa_error_domain_wine();
  }
  static inline constexpr ::std::size_t code(errc_type __e) noexcept {
    return static_cast<::std::size_t>(static_cast<::std::uint_least32_t>(__e));
  }
};

} // namespace std
