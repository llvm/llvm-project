// RUN: %check_clang_tidy -std=c++23 %s modernize-use-std-print %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             [ \
// RUN:              { \
// RUN:               key: modernize-use-std-print.PrintfLikeFunctions, \
// RUN:               value: 'unqualified_printf;::myprintf; mynamespace::myprintf2' \
// RUN:              }, \
// RUN:              { \
// RUN:               key: modernize-use-std-print.FprintfLikeFunctions, \
// RUN:               value: '::myfprintf; mynamespace::myfprintf2' \
// RUN:              } \
// RUN:             ] \
// RUN:            }" \
// RUN:   -- -isystem %clang_tidy_headers

#include <cstdio>
#include <string>

int myprintf(const char *, ...);
int myfprintf(FILE *fp, const char *, ...);

namespace mynamespace {
int myprintf2(const char *, ...);
int myfprintf2(FILE *fp, const char *, ...);
}

void printf_simple() {
  myprintf("Hello %s %d", "world", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'myprintf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("Hello {} {}", "world", 42);
}

void printf_newline() {
  myprintf("Hello %s %d\n", "world", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'myprintf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello {} {}", "world", 42);

  mynamespace::myprintf2("Hello %s %d\n", "world", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'myprintf2' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello {} {}", "world", 42);

  using mynamespace::myprintf2;
  myprintf2("Hello %s %d\n", "world", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'myprintf2' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello {} {}", "world", 42);

  // When using custom options leave printf alone
  printf("Hello %s %d\n", "world", 42);
}

int printf_uses_return_value(int i) {
  return myprintf("return value %d\n", i);
  // CHECK-MESSAGES-NOT: [[@LINE-1]]:10: warning: use 'std::println' instead of 'myprintf' [modernize-use-std-print]
  // CHECK-FIXES-NOT: std::println("return value {}", i);
}

void fprintf_simple(FILE *fp)
{
  myfprintf(stderr, "Hello %s %d", "world", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'myfprintf' [modernize-use-std-print]
  // CHECK-FIXES: std::print(stderr, "Hello {} {}", "world", 42);
}

void fprintf_newline(FILE *fp)
{
  myfprintf(stderr, "Hello %s %d\n", "world", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'myfprintf' [modernize-use-std-print]
  // CHECK-FIXES: std::println(stderr, "Hello {} {}", "world", 42);

  mynamespace::myfprintf2(stderr, "Hello %s %d\n", "world", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'myfprintf2' [modernize-use-std-print]
  // CHECK-FIXES: std::println(stderr, "Hello {} {}", "world", 42);

  using mynamespace::myfprintf2;
  myfprintf2(stderr, "Hello %s %d\n", "world", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'myfprintf2' [modernize-use-std-print]
  // CHECK-FIXES: std::println(stderr, "Hello {} {}", "world", 42);

  // When using custom options leave fprintf alone
  fprintf(stderr, "Hello %s %d\n", "world", 42);
}

int fprintf_uses_return_value(int i) {
  return myfprintf(stderr, "return value %d\n", i);
  // CHECK-MESSAGES-NOT: [[@LINE-1]]:10: warning: use 'std::println' instead of 'myprintf' [modernize-use-std-print]
  // CHECK-FIXES-NOT: std::println(stderr, "return value {}", i);
}

// Ensure that MatchesAnyListedNameMatcher::NameMatcher::match() can cope with a
// NamedDecl that has no name when we're trying to match unqualified_printf.
void no_name(const std::string &in)
{
  "A" + in;
}
