// RUN: %check_clang_tidy %s modernize-use-std-print %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             { \
// RUN:               modernize-use-std-print.ReplacementPrintFunction: 'fmt::print', \
// RUN:               modernize-use-std-print.ReplacementPrintlnFunction: 'fmt::println', \
// RUN:               modernize-use-std-print.PrintHeader: '<fmt/core.h>' \
// RUN:             } \
// RUN:            }" \
// RUN:   -- -isystem %clang_tidy_headers

#include <cstdio>
// CHECK-FIXES: #include <fmt/core.h>
#include <string.h>

void printf_simple() {
  printf("Hello %s %d", "world", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'fmt::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: fmt::print("Hello {} {}", "world", 42);
}

void printf_newline() {
  printf("Hello %s %d\n", "world", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'fmt::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: fmt::println("Hello {} {}", "world", 42);
}

void fprintf_simple(FILE *fp) {
  fprintf(fp, "Hello %s %d", "world", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'fmt::print' instead of 'fprintf' [modernize-use-std-print]
  // CHECK-FIXES: fmt::print(fp, "Hello {} {}", "world", 42);
}

void fprintf_newline(FILE *fp) {
  fprintf(fp, "Hello %s %d\n", "world", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'fmt::println' instead of 'fprintf' [modernize-use-std-print]
  // CHECK-FIXES: fmt::println(fp, "Hello {} {}", "world", 42);
}
