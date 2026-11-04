// RUN: %clang_cc1 -fsyntax-only -fblocks -verify -std=gnu11 -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -fblocks -verify -x c++ -std=gnu++98 -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -fblocks -std=gnu11 -Wmissing-format-attribute -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fblocks -x c++ -std=gnu++98 -Wmissing-format-attribute -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

typedef unsigned long size_t;
typedef long ssize_t;
typedef __builtin_va_list va_list;

__attribute__((format(printf, 1, 2)))
int printf(const char *, ...);

__attribute__((format(printf, 1, 0)))
int vprintf(const char *, va_list);

// Test that attribute fixit is specified using the GNU extension format when -std=gnuXY or -std=gnu++XY.

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"__attribute__((format(printf, 1, 0))) "
void f1(char *out, va_list args) // #f1
{
  vprintf(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 1, 0)' attribute to the declaration of 'f1'}}
                      // expected-note@#f1 {{'f1' declared here}}
}

void f2(void) {
  void (^b1)(const char *, ...) = ^(const char *fmt, ...) { // #b1
    va_list args;
    vprintf(fmt, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 1, 2)' attribute to the declaration of block}}
                        // expected-note@#b1 {{block declared here}}
  };

  void (^b2)(const char *, va_list) = ^(const char *fmt, va_list args) { // #b2
    vprintf(fmt, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 1, 0)' attribute to the declaration of block}}.
                        // expected-note@#b2 {{block declared here}}
  };

  void (^b3)(const char *, int x, float y) = ^(const char *fmt, int x, float y) { // #b3
    printf(fmt, x, y); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 1, 2)' attribute to the declaration of block}}.
                       // expected-note@#b3 {{block declared here}}
  };

  void __attribute__((__format__(__printf__, 1, 2))) (^b4)(const char *, ...) =
      ^(const char *fmt, ...) __attribute__((__format__(__printf__, 1, 2))) {
    va_list args;
    vprintf(fmt, args);
  };

  void __attribute__((__format__(__printf__, 2, 3))) (^b5)(const char *, const char *, ...) =
      ^(const char *not_fmt, const char *fmt, ...) __attribute__((__format__(__printf__, 2, 3))) { // #b5
    va_list args;
    vprintf(fmt, args);
    vprintf(not_fmt, args); // expected-warning{{diagnostic behavior may be improved by adding the 'format(printf, 1, 3)' attribute to the declaration of block}}
                            // expected-note@#b5 {{block declared here}}
  };
}
