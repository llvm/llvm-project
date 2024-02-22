//===-- OptionParsing.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_OPTION_PARSING_H
#define LLDB_UTILITY_OPTION_PARSING_H

#include "lldb/Utility/Args.h"

namespace lldb_private {
namespace OptionParsing {
inline bool GetOptionValueAsString(Args &args, const char *option,
                                   std::string &value) {
  for (size_t ai = 0, ae = args.GetArgumentCount(); ai != ae; ++ai) {
    const char *arg = args.GetArgumentAtIndex(ai);
    const char *option_loc = strstr(arg, option);

    const bool is_long_option = (option[0] == '-' && option[1] == '-');

    if (option_loc == arg) {
      const char *after_option = option_loc + strlen(option);

      switch (*after_option) {
      default:
        if (is_long_option) {
          continue;
        } else {
          value = after_option;
          return true;
        }
        break;
      case '=':
        value = after_option + 1;
        return true;
      case '\0': {
        const char *next_value = args.GetArgumentAtIndex(ai + 1);
        if (next_value) {
          value = next_value;
          return true;
        } else {
          return false;
        }
      }
      }
    }
  }

  return false;
}

inline int GetOptionValuesAsStrings(Args &args, const char *option,
                                    std::vector<std::string> &values) {
  int ret = 0;

  for (size_t ai = 0, ae = args.GetArgumentCount(); ai != ae; ++ai) {
    const char *arg = args.GetArgumentAtIndex(ai);
    const char *option_loc = strstr(arg, option);

    const bool is_long_option = (option[0] == '-' && option[1] == '-');

    if (option_loc == arg) {
      const char *after_option = option_loc + strlen(option);

      switch (*after_option) {
      default:
        if (is_long_option) {
          continue;
        } else {
          values.push_back(after_option);
          ++ret;
        }
        break;
      case '=':
        values.push_back(after_option + 1);
        ++ret;
        break;
      case '\0': {
        const char *next_value = args.GetArgumentAtIndex(ai + 1);
        if (next_value) {
          values.push_back(next_value);
          ++ret;
          ++ai;
          break;
        } else {
          return ret;
        }
      }
      }
    }
  }

  return ret;
}

} // namespace OptionParsing
} // namespace lldb_private

#endif // LLDB_UTILITY_OPTION_PARSING_H
