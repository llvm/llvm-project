// RUN: %clang_cc1 -fsyntax-only -verify -std=gnu11 -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -verify -x c++ -std=gnu++98 -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -std=gnu11 -Wmissing-format-attribute -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -x c++ -std=gnu++98 -Wmissing-format-attribute -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

typedef unsigned long size_t;
typedef long ssize_t;
typedef __builtin_va_list va_list;

__attribute__((format(printf, 1, 0)))
int vprintf(const char *, va_list);

// Test that attribute fixit is specified using the GNU extension format when -std=gnuXY or -std=gnu++XY.

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"__attribute__((format(printf, 1, 0))) "
void f1(char *out, va_list args) // #f1
{
  vprintf(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 1, 0)' attribute to the declaration of 'f1'}}
                      // expected-note@#f1 {{'f1' declared here}}
}
