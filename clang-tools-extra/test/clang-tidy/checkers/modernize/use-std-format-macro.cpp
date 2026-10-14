// RUN: %check_clang_tidy -std=c++20-or-later %s modernize-use-std-format %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:              modernize-use-std-format.StrFormatLikeFunctions: 'String::Printf', \
// RUN:              modernize-use-std-format.ReplacementFormatFunction: 'fmt::format', \
// RUN:              modernize-use-std-format.FormatHeader: '<fmt/format.h>' \
// RUN:            }}"

#include <string>
// CHECK-FIXES: #include <fmt/format.h>

namespace String {
std::string Printf(const char *format, ...);
} // namespace String

#define WRAP_MSG(msg) msg

std::string macro_argument_include(int n) {
  return WRAP_MSG(String::Printf("value %d", n));
  // CHECK-MESSAGES: [[@LINE-1]]:19: warning: use 'fmt::format' instead of 'Printf' [modernize-use-std-format]
  // CHECK-FIXES: return WRAP_MSG(fmt::format("value {}", n));
}
