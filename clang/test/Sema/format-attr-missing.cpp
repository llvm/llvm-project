// RUN: %clang_cc1 -fsyntax-only -verify -std=c++23 -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -Wmissing-format-attribute -fdiagnostics-parseable-fixits -std=c++23 %s 2>&1 | FileCheck %s

typedef __SIZE_TYPE__ size_t;
typedef __builtin_va_list va_list;

[[gnu::format(printf, 1, 2)]]
int printf(const char *, ...);

[[gnu::format(scanf, 1, 2)]]
int scanf(const char *, ...);

[[gnu::format(printf, 1, 0)]]
int vprintf(const char *, va_list);

[[gnu::format(scanf, 1, 0)]]
int vscanf(const char *, va_list);

[[gnu::format(printf, 2, 0)]]
int vsprintf(char *, const char *, va_list);

[[gnu::format(printf, 3, 0)]]
int vsnprintf(char *, size_t, const char *, va_list);

struct S1
{
  // CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:3-[[@LINE+1]]:3}:"{{\[\[}}gnu::format(scanf, 2, 3)]] "
  void fn1(const char *out, ... /* args */) // #S1_fn1
  {
    va_list args;
    vscanf(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(scanf, 2, 3)' attribute to the declaration of 'fn1'}}
                       // expected-note@#S1_fn1 {{'fn1' declared here}}
  }

  [[gnu::format(printf, 2, 0)]]
  void print(const char *out, va_list args);

  // CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:3-[[@LINE+1]]:3}:"{{\[\[}}gnu::format(printf, 2, 3)]] "
  void fn2(const char *out, ... /* args */) // #S1_fn2
  {
    va_list args;
    print(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 2, 3)' attribute to the declaration of 'fn2'}}
                      // expected-note@#S1_fn2 {{'fn2' declared here}}
  }

  // CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:3-[[@LINE+1]]:3}:"{{\[\[}}gnu::format(printf, 2, 0)]] "
  void fn3(const char *out, va_list args) // #S1_fn3
  {
    print(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 2, 0)' attribute to the declaration of 'fn3'}}
                      // expected-note@#S1_fn3 {{'fn3' declared here}}
  }

  // CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:3-[[@LINE+1]]:3}:"{{\[\[}}gnu::format(printf, 2, 3)]] "
  void fn4(this S1& self, const char *out, ... /* args */) // #S1_fn4
  {
    va_list args;
    self.print(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 2, 3)' attribute to the declaration of 'fn4'}}
                           // expected-note@#S1_fn4 {{'fn4' declared here}}
  }

  // CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:3-[[@LINE+1]]:3}:"{{\[\[}}gnu::format(printf, 2, 0)]] "
  void fn5(this S1& self, const char *out, va_list args) // #S1_fn5
  {
    self.print(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 2, 0)' attribute to the declaration of 'fn5'}}
                           // expected-note@#S1_fn5 {{'fn5' declared here}}
  }
};

