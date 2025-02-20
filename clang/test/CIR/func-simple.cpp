// Simple functions
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o -  | FileCheck %s

void empty() { }
// CHECK: cir.func @empty() -> !cir.void {
// CHECK:   cir.return
// CHECK: }

void voidret() { return; }
// CHECK: cir.func @voidret() -> !cir.void {
// CHECK:   cir.return
// CHECK: }

int intfunc() { return 42; }
// CHECK: cir.func @intfunc() -> !cir.int<s, 32> {
// CHECK:   %0 = cir.const #cir.int<42> : !cir.int<s, 32>
// CHECK:   cir.return %0 : !cir.int<s, 32>
// CHECK: }

int scopes() {
  {
    {
      return 99;
    }
  }
}
// CHECK: cir.func @scopes() -> !cir.int<s, 32> {
// CHECK:   cir.scope {
// CHECK:     cir.scope {
// CHECK:       %0 = cir.const #cir.int<99> : !cir.int<s, 32>
// CHECK:       cir.return %0 : !cir.int<s, 32>
// CHECK:     }
// CHECK:   }
// CHECK:   cir.trap
// CHECK: }

long longfunc() { return 42l; }
// CHECK: cir.func @longfunc() -> !cir.int<s, 64>
// CHECK:   %0 = cir.const #cir.int<42> : !cir.int<s, 64>
// CHECK:   cir.return %0 : !cir.int<s, 64>
// CHECK: }

unsigned unsignedfunc() { return 42u; }
// CHECK: cir.func @unsignedfunc() -> !cir.int<u, 32>
// CHECK:   %0 = cir.const #cir.int<42> : !cir.int<u, 32>
// CHECK:   cir.return %0 : !cir.int<u, 32>
// CHECK: }

unsigned long long ullfunc() { return 42ull; }
// CHECK: cir.func @ullfunc() -> !cir.int<u, 64>
// CHECK:   %0 = cir.const #cir.int<42> : !cir.int<u, 64>
// CHECK:   cir.return %0 : !cir.int<u, 64>
// CHECK: }
