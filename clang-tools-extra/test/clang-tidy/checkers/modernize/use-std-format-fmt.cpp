// RUN: %check_clang_tidy %s modernize-use-std-format %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             [ \
// RUN:              { \
// RUN:                key: StrictMode, value: true \
// RUN:              }, \
// RUN:              { \
// RUN:               key: modernize-use-std-format.StrFormatLikeFunctions, \
// RUN:               value: 'fmt::sprintf' \
// RUN:              }, \
// RUN:              { \
// RUN:               key: modernize-use-std-format.ReplacementFormatFunction, \
// RUN:               value: 'fmt::format' \
// RUN:              }, \
// RUN:              { \
// RUN:               key: modernize-use-std-format.FormatHeader, \
// RUN:               value: '<fmt/core.h>' \
// RUN:              } \
// RUN:             ] \
// RUN:            }" \
// RUN:   -- -isystem %clang_tidy_headers

// CHECK-FIXES: #include <fmt/core.h>
#include <string>

namespace fmt
{
// Use const char * for the format since the real type is hard to mock up.
template <typename... Args>
std::string sprintf(const char *format, const Args&... args);
} // namespace fmt

std::string fmt_sprintf_simple() {
  return fmt::sprintf("Hello %s %d", "world", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: use 'fmt::format' instead of 'sprintf' [modernize-use-std-format]
  // CHECK-FIXES: fmt::format("Hello {} {}", "world", 42);
}
