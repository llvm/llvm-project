// RUN: %check_clang_tidy -check-suffixes=,STRICT \
// RUN:   -std=c++20 %s modernize-use-std-format %t --      \
// RUN:   -config="{CheckOptions: \
// RUN:             [ \
// RUN:              { \
// RUN:                key: StrictMode, value: true \
// RUN:              }, \
// RUN:              { \
// RUN:               key: modernize-use-std-format.StrFormatLikeFunctions, \
// RUN:               value: 'unqualified_strprintf;::strprintf; mynamespace::strprintf2' \
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
// RUN: %check_clang_tidy -check-suffixes=,NOTSTRICT \
// RUN:   -std=c++20 %s modernize-use-std-format %t --      \
// RUN:   -config="{CheckOptions: \
// RUN:             [ \
// RUN:              { \
// RUN:               key: modernize-use-std-format.StrFormatLikeFunctions, \
// RUN:               value: '::strprintf; mynamespace::strprintf2' \
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

#include <cstdio>
#include <string>
// CHECK-FIXES: #include <fmt/core.h>

std::string strprintf(const char *, ...);

namespace mynamespace {
  std::string strprintf2(const char *, ...);
}

std::string strprintf_test(const std::string &name, double value) {
  return strprintf("'%s'='%f'\n", name.c_str(), value);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: use 'fmt::format' instead of 'strprintf' [modernize-use-std-format]
  // CHECK-FIXES: return fmt::format("'{}'='{:f}'\n", name, value);

  return mynamespace::strprintf2("'%s'='%f'\n", name.c_str(), value);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: use 'fmt::format' instead of 'strprintf2' [modernize-use-std-format]
  // CHECK-FIXES: return fmt::format("'{}'='{:f}'\n", name, value);
}

std::string StrFormat_strict_conversion() {
  const unsigned char uc = 'A';
  return strprintf("Integer %hhd from unsigned char\n", uc);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: use 'fmt::format' instead of 'strprintf' [modernize-use-std-format]
  // CHECK-FIXES-NOTSTRICT: return fmt::format("Integer {} from unsigned char\n", uc);
  // CHECK-FIXES-STRICT: return fmt::format("Integer {} from unsigned char\n", static_cast<signed char>(uc));
}

// Ensure that MatchesAnyListedNameMatcher::NameMatcher::match() can cope with a
// NamedDecl that has no name when we're trying to match unqualified_strprintf.
std::string A(const std::string &in)
{
    return "_" + in;
}
