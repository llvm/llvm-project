// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___DEBUGGING_SUPPORT_LINUX_H
#define _LIBCPP___DEBUGGING_SUPPORT_LINUX_H

#include <__config>
#include <array>
#include <fcntl.h>
#include <string_view>
#include <unistd.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26

_LIBCPP_HIDE_FROM_ABI inline bool __libcpp_is_debugger_present() noexcept {
  // https://docs.kernel.org/filesystems/proc.html
  alignas(8) array<char, 256 + 1> __buffer{};
  constexpr std::string_view __tracer_key("\nTracerPid:\t"); // Linux >= 2.6.0

  int __buf_read      = ::open("/proc/self/status", O_RDONLY | O_CLOEXEC);
  const auto __result = ::read(__buf_read, __buffer.data(), __buffer.size() - 1);
  ::close(__buf_read);

  if (__result < 80) {
    return false;
  }

  // skip process name block
  std::string_view __view(__buffer.data() + 64, __result - 64);
  auto __tracerpid = __view.find(__tracer_key);

  if (__tracerpid == std::string_view::npos) {
    return false;
  }

  __view.remove_prefix(__tracerpid);         // Remove everything upto \nTracerPid:\t
  __view.remove_prefix(__tracer_key.size()); // remove \nTracerPid:\t

  const auto __pidn = __view.find('\n');
  if (__pidn == std::string_view::npos) {
    return false;
  }

  const std::string_view __pid = __view.substr(0, __pidn); // remove '\n'

  if (__pid[0] == '0') {
    return false;
  }

  auto __copied = std::string_view("/proc/").copy(__buffer.data(), __buffer.size());
  __copied += __pid.copy(__buffer.data() + __copied, __buffer.size() - __copied);
  __copied += std::string_view("/comm").copy(__buffer.data() + __copied, __buffer.size() - __copied);
  __buffer[__copied] = '\0';

  const int __tracer_read = ::open(__buffer.data(), O_RDONLY | O_CLOEXEC); // Linux >= 2.6.33
  // https://elixir.bootlin.com/linux/latest/source/include/linux/sched.h#L325
  const auto __tracer_result = ::read(__tracer_read, __buffer.data(), 16 + 1);
  ::close(__tracer_read);

  if (__tracer_result <= 0) {
    return false;
  }

  __buffer[__tracer_result - 1] = '\0'; // remove newline from /proc/xyz/comm content
  const std::string_view __tracer_name(__buffer.data());

  // Sniff for known debuggers
  for (auto __i : {"gdb", "gdbserver", "lldb-server"}) {
    if (__tracer_name.starts_with(__i)) {
      return true;
    }
  }

  return false;
}

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___DEBUGGING_SUPPORT_LINUX_H
