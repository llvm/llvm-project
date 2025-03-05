// Simple functions
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o -  | FileCheck %s

void empty() { }
// CHECK: cir.func @empty() {
// CHECK:   cir.return
// CHECK: }

void voidret() { return; }
// CHECK: cir.func @voidret() {
// CHECK:   cir.return
// CHECK: }

int intfunc() { return 42; }
// CHECK: cir.func @intfunc() -> !cir.int<s, 32> {
// CHECK:   %0 = cir.alloca !cir.int<s, 32>, !cir.ptr<!cir.int<s, 32>>, ["__retval"] {alignment = 4 : i64}
// CHECK:   %1 = cir.const #cir.int<42> : !cir.int<s, 32>
// CHECK:   cir.return %1 : !cir.int<s, 32>
// CHECK: }

int scopes() {
  {
    {
      return 99;
    }
  }
}
// CHECK: cir.func @scopes() -> !cir.int<s, 32> {
// CHECK:   %0 = cir.alloca !cir.int<s, 32>, !cir.ptr<!cir.int<s, 32>>, ["__retval"] {alignment = 4 : i64}
// CHECK:   cir.scope {
// CHECK:     cir.scope {
// CHECK:       %1 = cir.const #cir.int<99> : !cir.int<s, 32>
// CHECK:       cir.return %1 : !cir.int<s, 32>
// CHECK:     }
// CHECK:   }
// CHECK:   cir.trap
// CHECK: }

long longfunc() { return 42l; }
// CHECK: cir.func @longfunc() -> !cir.int<s, 64>
// CHECK:   %0 = cir.alloca !cir.int<s, 64>, !cir.ptr<!cir.int<s, 64>>, ["__retval"] {alignment = 8 : i64}
// CHECK:   %1 = cir.const #cir.int<42> : !cir.int<s, 64>
// CHECK:   cir.return %1 : !cir.int<s, 64>
// CHECK: }

unsigned unsignedfunc() { return 42u; }
// CHECK: cir.func @unsignedfunc() -> !cir.int<u, 32>
// CHECK:   %0 = cir.alloca !cir.int<u, 32>, !cir.ptr<!cir.int<u, 32>>, ["__retval"] {alignment = 4 : i64}
// CHECK:   %1 = cir.const #cir.int<42> : !cir.int<u, 32>
// CHECK:   cir.return %1 : !cir.int<u, 32>
// CHECK: }

unsigned long long ullfunc() { return 42ull; }
// CHECK: cir.func @ullfunc() -> !cir.int<u, 64>
// CHECK:   %0 = cir.alloca !cir.int<u, 64>, !cir.ptr<!cir.int<u, 64>>, ["__retval"] {alignment = 8 : i64}
// CHECK:   %1 = cir.const #cir.int<42> : !cir.int<u, 64>
// CHECK:   cir.return %1 : !cir.int<u, 64>
// CHECK: }

bool boolfunc() { return true; }
// CHECK: cir.func @boolfunc() -> !cir.bool {
// CHECK:   %0 = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["__retval"] {alignment = 1 : i64}
// CHECK:   %1 = cir.const #true
// CHECK:   cir.return %1 : !cir.bool
// CHECK: }

float floatfunc() { return 42.42f; }
// CHECK: cir.func @floatfunc() -> !cir.float {
// CHECK:   %0 = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["__retval"] {alignment = 4 : i64}
// CHECK:   %1 = cir.const #cir.fp<4.242
// CHECK:   cir.return %1 : !cir.float
// CHECK: }

double doublefunc() { return 42.42; }
// CHECK: cir.func @doublefunc() -> !cir.double {
// CHECK:   %0 = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["__retval"] {alignment = 8 : i64}
// CHECK:   %1 = cir.const #cir.fp<4.242
// CHECK:   cir.return %1 : !cir.double
// CHECK: }
