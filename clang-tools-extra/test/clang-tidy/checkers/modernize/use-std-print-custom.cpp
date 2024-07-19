// RUN: %check_clang_tidy -std=c++23 %s modernize-use-std-print %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             { \
// RUN:               modernize-use-std-print.PrintfLikeFunctions: 'unqualified_printf;::myprintf; mynamespace::myprintf2; bad_format_type_printf; fmt::printf', \
// RUN:               modernize-use-std-print.FprintfLikeFunctions: '::myfprintf; mynamespace::myfprintf2; bad_format_type_fprintf; fmt::fprintf' \
// RUN:             } \
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

int myprintf(const wchar_t *, ...);

void wide_string_not_supported() {
  myprintf(L"wide string %s", L"string");
}

// Issue #92896: Ensure that the check doesn't assert if the argument is
// promoted to something that isn't a string.
struct S {
  S(...) {}
};
int bad_format_type_printf(const S &, ...);
int bad_format_type_fprintf(FILE *, const S &, ...);

void unsupported_format_parameter_type()
{
  // No fixes here because the format parameter of the function called is not a
  // string.
  bad_format_type_printf("Hello %s", "world");
// CHECK-MESSAGES: [[@LINE-1]]:3: warning: unable to use 'std::print' instead of 'bad_format_type_printf' because first argument is not a narrow string literal [modernize-use-std-print]

  bad_format_type_fprintf(stderr, "Hello %s", "world");
// CHECK-MESSAGES: [[@LINE-1]]:3: warning: unable to use 'std::print' instead of 'bad_format_type_fprintf' because first argument is not a narrow string literal [modernize-use-std-print]
}

namespace fmt {
  template <typename S, typename... T>
  inline int printf(const S& fmt, const T&... args);

  template <typename S, typename... T>
  inline int fprintf(std::FILE* f, const S& fmt, const T&... args);
}

void fmt_printf()
{
  fmt::printf("fmt::printf templated %s argument %d\n", "format", 424);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("fmt::printf templated {} argument {}", "format", 424);

  fmt::fprintf(stderr, "fmt::fprintf templated %s argument %d\n", "format", 425);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
  // CHECK-FIXES: std::println(stderr, "fmt::fprintf templated {} argument {}", "format", 425);
}
