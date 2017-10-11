// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -E %s | FileCheck %s

// CHECK: #pragma clang assume_nonnull begin
#pragma clang assume_nonnull begin

int bar(int * ip) { return *ip; }

// CHECK: #pragma clang assume_nonnull end
#pragma clang assume_nonnull end

int foo(int * _Nonnull ip) { return *ip; }

int main() {
   return bar(0) + foo(0); // expected-warning 2 {{null passed to a callee that requires a non-null argument}}
}
