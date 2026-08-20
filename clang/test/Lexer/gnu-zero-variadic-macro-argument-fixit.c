// RUN: %clang_cc1 -fsyntax-only -verify %s -Wgnu-zero-variadic-macro-arguments -std=c23
// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits %s -Wgnu-zero-variadic-macro-arguments -std=c23 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -verify %s -Wgnu-zero-variadic-macro-arguments -xc++ -std=c++20
// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits %s -Wgnu-zero-variadic-macro-arguments -xc++ -std=c++20 2>&1 | FileCheck %s

void foo(const char* fmt, ...);
// CHECK: fix-it:"{{.*}}":{[[@LINE+2]]:36-[[@LINE+2]]:51}:" __VA_OPT__(,) __VA_ARGS__"
// expected-warning@+1 {{token pasting of ',' and '__VA_ARGS__' is a GNU extension; consider using '__VA_OPT__(,)' instead}}
#define FOO(format, ...) foo(format, ##__VA_ARGS__)

void bar(void) {
  FOO("", 0);
}
