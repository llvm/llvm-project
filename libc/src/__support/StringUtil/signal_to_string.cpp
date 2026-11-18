//===-- Implementation of a class for mapping signals to strings ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "signal_to_string.h"

#include <stddef.h>

#include "hdr/signal_macros.h"
#include "platform_signals.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/stringstream.h"
#include "src/__support/StringUtil/message_mapper.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/integer_to_string.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace {

constexpr size_t max_buff_size() {
  constexpr size_t base_str_len = sizeof("Real-time signal");
  // the buffer should be able to hold "Real-time signal" + ' ' + num_str
  return (base_str_len + 1 + IntegerToString<int>::buffer_size()) *
         sizeof(char);
}

// This is to hold signal strings that have to be custom built. It may be
// rewritten on every call to strsignal (or other signal to string function).
constexpr size_t SIG_BUFFER_SIZE = max_buff_size();
LIBC_THREAD_LOCAL char signal_buffer[SIG_BUFFER_SIZE];

constexpr size_t TOTAL_STR_LEN = total_str_len(PLATFORM_SIGNALS);

constexpr size_t SIG_ARRAY_SIZE = max_key_val(PLATFORM_SIGNALS) + 1;

constexpr MessageMapper<SIG_ARRAY_SIZE, TOTAL_STR_LEN>
    signal_mapper(PLATFORM_SIGNALS);

cpp::string_view build_signal_string(int sig_num, cpp::span<char> buffer) {
  cpp::string_view base_str;
  if (sig_num >= SIGRTMIN && sig_num <= SIGRTMAX) {
    base_str = cpp::string_view("Real-time signal");
    sig_num -= SIGRTMIN;
  } else {
    base_str = cpp::string_view("Unknown signal");
  }

  // if the buffer can't hold "Unknown signal" + ' ' + num_str, then just
  // return "Unknown signal".
  if (buffer.size() <
      (base_str.size() + 1 + IntegerToString<int>::buffer_size()))
    return base_str;

  cpp::StringStream buffer_stream(
      {const_cast<char *>(buffer.data()), buffer.size()});
  buffer_stream << base_str << ' ' << sig_num << '\0';
  return buffer_stream.str();
}

constexpr size_t SIG_NAME_BUFFER_SIZE = 32;
LIBC_THREAD_LOCAL char signal_name_buffer[SIG_NAME_BUFFER_SIZE];

constexpr size_t TOTAL_NAME_STR_LEN = total_str_len(PLATFORM_SIGNAL_NAMES);
constexpr size_t SIG_NAME_ARRAY_SIZE = max_key_val(PLATFORM_SIGNAL_NAMES) + 1;

constexpr MessageMapper<SIG_NAME_ARRAY_SIZE, TOTAL_NAME_STR_LEN>
    signal_name_mapper(PLATFORM_SIGNAL_NAMES);

cpp::string_view build_signal_name(int sig_num, cpp::span<char> buffer) {
#if defined(SIGRTMIN) && defined(SIGRTMAX)
  if (sig_num >= SIGRTMIN && sig_num <= SIGRTMAX) {
    if (sig_num == SIGRTMIN)
      return "RTMIN";
    if (sig_num == SIGRTMAX)
      return "RTMAX";

    if (buffer.size() < sizeof("RTMIN+999"))
      return cpp::string_view();

    cpp::StringStream buffer_stream(
        {const_cast<char *>(buffer.data()), buffer.size()});
    if (sig_num <= SIGRTMIN + (SIGRTMAX - SIGRTMIN) / 2) {
      buffer_stream << "RTMIN+" << (sig_num - SIGRTMIN) << '\0';
    } else {
      buffer_stream << "RTMAX-" << (SIGRTMAX - sig_num) << '\0';
    }
    return buffer_stream.str();
  }
#endif
  return cpp::string_view();
}

} // namespace

cpp::string_view get_signal_string(int sig_num) {
  return get_signal_string(sig_num, {signal_buffer, SIG_BUFFER_SIZE});
}

cpp::string_view get_signal_string(int sig_num, cpp::span<char> buffer) {
  auto opt_str = signal_mapper.get_str(sig_num);
  if (opt_str)
    return *opt_str;
  else
    return build_signal_string(sig_num, buffer);
}

cpp::string_view get_signal_name(int sig_num) {
  return get_signal_name(sig_num, {signal_name_buffer, SIG_NAME_BUFFER_SIZE});
}

cpp::string_view get_signal_name(int sig_num, cpp::span<char> buffer) {
  auto opt_str = signal_name_mapper.get_str(sig_num);
  if (opt_str && !opt_str->empty())
    return *opt_str;
  return build_signal_name(sig_num, buffer);
}

cpp::optional<int> get_signal_number(cpp::string_view sig_name) {
  if (sig_name.empty())
    return cpp::nullopt;

  if (sig_name == "0")
    return 0;

  for (size_t i = 0; i < PLATFORM_SIGNAL_NAMES.size(); ++i) {
    if (PLATFORM_SIGNAL_NAMES[i].msg == sig_name &&
        PLATFORM_SIGNAL_NAMES[i].num != 0)
      return PLATFORM_SIGNAL_NAMES[i].num;
  }

#if defined(SIGRTMIN) && defined(SIGRTMAX)
  if (sig_name == "RTMIN")
    return SIGRTMIN;
  if (sig_name == "RTMAX")
    return SIGRTMAX;

  if (sig_name.starts_with("RTMIN+")) {
    cpp::string_view num_str = sig_name.substr(sizeof("RTMIN+") - 1);
    if (!num_str.empty()) {
      int offset = 0;
      bool valid = true;
      for (size_t i = 0; i < num_str.size(); ++i) {
        if (!internal::isdigit(num_str[i])) {
          valid = false;
          break;
        }
        int d = num_str[i] - '0';
        if (offset > (2147483647 - d) / 10) {
          valid = false;
          break;
        }
        offset = offset * 10 + d;
      }
      if (valid && offset >= 0 && offset <= (SIGRTMAX - SIGRTMIN) / 2)
        return SIGRTMIN + offset;
    }
  }

  if (sig_name.starts_with("RTMAX-")) {
    cpp::string_view num_str = sig_name.substr(sizeof("RTMAX-") - 1);
    if (!num_str.empty()) {
      int offset = 0;
      bool valid = true;
      for (size_t i = 0; i < num_str.size(); ++i) {
        if (!internal::isdigit(num_str[i])) {
          valid = false;
          break;
        }
        int d = num_str[i] - '0';
        if (offset > (2147483647 - d) / 10) {
          valid = false;
          break;
        }
        offset = offset * 10 + d;
      }
      if (valid && offset >= 0 && offset <= (SIGRTMAX - SIGRTMIN) / 2)
        return SIGRTMAX - offset;
    }
  }
#endif

  // Check if sig_name is a valid integer representation of a signal number.
  bool all_digits = true;
  for (size_t i = 0; i < sig_name.size(); ++i) {
    if (!internal::isdigit(sig_name[i])) {
      all_digits = false;
      break;
    }
  }
  if (all_digits) {
    int num = 0;
    bool overflow = false;
    for (size_t i = 0; i < sig_name.size(); ++i) {
      int d = sig_name[i] - '0';
      if (num > (2147483647 - d) / 10) {
        overflow = true;
        break;
      }
      num = num * 10 + d;
    }
    if (!overflow && num > 0) {
#ifdef NSIG
      if (num < NSIG) {
        char buf[SIG_NAME_BUFFER_SIZE];
        if (!get_signal_name(num, {buf, sizeof(buf)}).empty())
          return num;
      }
#else
      char buf[SIG_NAME_BUFFER_SIZE];
      if (!get_signal_name(num, {buf, sizeof(buf)}).empty())
        return num;
#endif
    }
  }

  return cpp::nullopt;
}

} // namespace LIBC_NAMESPACE_DECL
