// RUN: %check_clang_tidy \
// RUN:   -std=c++23 %s modernize-use-std-print %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-std-print.StrictMode: true}}" \
// RUN:   -- -isystem %clang_tidy_headers
// RUN: %check_clang_tidy \
// RUN:   -std=c++23 %s modernize-use-std-print %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-std-print.StrictMode: false}}" \
// RUN:   -- -isystem %clang_tidy_headers

#include <cstdio>
#include <string.h>

namespace absl
{
// Use const char * for the format since the real type is hard to mock up.
template <typename... Args>
int PrintF(const char *format, const Args&... args);

template <typename... Args>
int FPrintF(FILE* output, const char *format, const Args&... args);
}

void printf_simple() {
  absl::PrintF("Hello %s %d", "world", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'PrintF' [modernize-use-std-print]
  // CHECK-FIXES: std::print("Hello {} {}", "world", 42);
}

void printf_newline() {
  absl::PrintF("Hello %s %d\n", "world", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'PrintF' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello {} {}", "world", 42);

  using namespace absl;
  PrintF("Hello %s %d\n", "world", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'PrintF' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello {} {}", "world", 42);
}

// absl uses the type of the argument rather than the format string, so unsigned
// types will be printed as unsigned even if the format string indicates signed
// and vice-versa. This is exactly what std::print will do too, so no casts are
// required.
void printf_no_casts_in_strict_mode() {
  using namespace absl;

  const unsigned short us = 42U;
  PrintF("Integer %hd from unsigned short\n", us);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'PrintF' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Integer {} from unsigned short", us);

  const short s = 42;
  PrintF("Unsigned integer %hu from short\n", s);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'PrintF' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Unsigned integer {} from short", s);
}

int printf_uses_return_value(int i) {
  using namespace absl;

  return PrintF("return value %d\n", i);
}

void fprintf_simple(FILE *fp) {
  absl::FPrintF(fp, "Hello %s %d", "world", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'FPrintF' [modernize-use-std-print]
  // CHECK-FIXES: std::print(fp, "Hello {} {}", "world", 42);
}

void fprintf_newline(FILE *fp) {
  absl::FPrintF(fp, "Hello %s %d\n", "world", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'FPrintF' [modernize-use-std-print]
  // CHECK-FIXES: std::println(fp, "Hello {} {}", "world", 42);

  using namespace absl;
  FPrintF(fp, "Hello %s %d\n", "world", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'FPrintF' [modernize-use-std-print]
  // CHECK-FIXES: std::println(fp, "Hello {} {}", "world", 42);
}

void fprintf_no_casts_in_strict_mode(FILE *fp) {
  using namespace absl;

  const unsigned short us = 42U;
  FPrintF(fp, "Integer %hd from unsigned short\n", us);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'FPrintF' [modernize-use-std-print]
  // CHECK-FIXES: std::println(fp, "Integer {} from unsigned short", us);

  const short s = 42;
  FPrintF(fp, "Unsigned integer %hu from short\n", s);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'FPrintF' [modernize-use-std-print]
  // CHECK-FIXES: std::println(fp, "Unsigned integer {} from short", s);
}

int fprintf_uses_return_value(FILE *fp, int i) {
  using namespace absl;

  return FPrintF(fp, "return value %d\n", i);
}
