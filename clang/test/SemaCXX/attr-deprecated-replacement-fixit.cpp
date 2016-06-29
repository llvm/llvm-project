// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: cp %s %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fixit %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -Werror %t

#if !__has_feature(attribute_deprecated_with_replacement)
#error "Missing __has_feature"
#endif

#if !__has_feature(attribute_availability_with_replacement)
#error "Missing __has_feature"
#endif

void f_8(int) __attribute__((deprecated("message", "new8"))); // expected-note {{'f_8' has been explicitly marked deprecated here}}
void new8(int);
void f_2(int) __attribute__((availability(macosx,deprecated=9.0,replacement="new2"))); // expected-note {{'f_2' has been explicitly marked deprecated here}}
void new2(int);
void test() {
  f_8(0); // expected-warning{{'f_8' is deprecated}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:6}:"new8"
  f_2(0); // expected-warning{{'f_2' is deprecated: first deprecated in macOS 9.0}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:6}:"new2"
}
