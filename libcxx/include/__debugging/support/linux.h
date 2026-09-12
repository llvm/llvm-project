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

#include <__algorithm/ranges_find_if.h>
#include <__config>
#include <__ranges/split_view.h>
#include <array>
#include <fcntl.h>
#include <string>
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
  constexpr std::string_view __tracer_key("TracerPid:\t"); // Linux >= 2.6.0

  int __buf_read      = ::open("/proc/self/status", O_RDONLY | O_CLOEXEC);
  const auto __result = ::read(__buf_read, __buffer.data(), __buffer.size() - 1);
  ::close(__buf_read);

  if (__result < 80) {
    return false;
  }

  const std::string_view __view(__buffer.data(), __result);

  auto __split           = std::ranges::views::split(__view, '\n');
  auto __tracer_pid_line = std::ranges::find_if(__split, [&](const auto __line) {
    return std::string_view(__line).starts_with(__tracer_key);
  });

  if (__tracer_pid_line == __split.end()) {
    return false;
  }

  std::string_view __pid(*__tracer_pid_line);
  __pid.remove_prefix(__tracer_key.size()); // remove "TracerPid:\t"

  if (__pid[0] == '0') {
    return false;
  }

  // https://elixir.bootlin.com/linux/latest/source/include/linux/sched.h#L325
  std::array<char, 16> __name_buffer{};

  const std::string __to_open = "/proc/"s + __pid + "/comm"sv;
  int __tracer_read           = ::open(__to_open.c_str(), O_RDONLY | O_CLOEXEC); // Linux >= 2.6.33
  const auto __tracer_result  = ::read(__tracer_read, __name_buffer.data(), __name_buffer.size());
  ::close(__tracer_read);

  if (__tracer_result < 0) {
    return false;
  }

  __name_buffer[__tracer_result - 1] = '\0'; // remove newline
  const std::string_view __tracer_name(__name_buffer.data());

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
