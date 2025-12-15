// Simple functions
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o -  | FileCheck %s

void empty() { }
// CHECK: cir.func{{.*}} @_Z5emptyv()
// CHECK:   cir.return
// CHECK: }

void voidret() { return; }
// CHECK: cir.func{{.*}} @_Z7voidretv()
// CHECK:   cir.return
// CHECK: }

int intfunc() { return 42; }
// CHECK: cir.func{{.*}} @_Z7intfuncv() -> !s32i
// CHECK:   %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:   %1 = cir.const #cir.int<42> : !s32i
// CHECK:   cir.store %1, %0 : !s32i, !cir.ptr<!s32i>
// CHECK:   %2 = cir.load %0 : !cir.ptr<!s32i>, !s32i
// CHECK:   cir.return %2 : !s32i
// CHECK: }

int scopes() {
  {
    {
      return 99;
    }
  }
}
// CHECK: cir.func{{.*}} @_Z6scopesv() -> !s32i
// CHECK:   %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:   cir.scope {
// CHECK:     cir.scope {
// CHECK:       %1 = cir.const #cir.int<99> : !s32i
// CHECK:       cir.store %1, %0 : !s32i, !cir.ptr<!s32i>
// CHECK:       %2 = cir.load %0 : !cir.ptr<!s32i>, !s32i
// CHECK:       cir.return %2 : !s32i
// CHECK:     }
// CHECK:   }
// CHECK:   cir.trap
// CHECK: }

long longfunc() { return 42l; }
// CHECK: cir.func{{.*}} @_Z8longfuncv() -> !s64i
// CHECK:   %0 = cir.alloca !s64i, !cir.ptr<!s64i>, ["__retval"] {alignment = 8 : i64}
// CHECK:   %1 = cir.const #cir.int<42> : !s64i
// CHECK:   cir.store %1, %0 : !s64i, !cir.ptr<!s64i>
// CHECK:   %2 = cir.load %0 : !cir.ptr<!s64i>, !s64i
// CHECK:   cir.return %2 : !s64i
// CHECK: }

unsigned unsignedfunc() { return 42u; }
// CHECK: cir.func{{.*}} @_Z12unsignedfuncv() -> !u32i
// CHECK:   %0 = cir.alloca !u32i, !cir.ptr<!u32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:   %1 = cir.const #cir.int<42> : !u32i
// CHECK:   cir.store %1, %0 : !u32i, !cir.ptr<!u32i>
// CHECK:   %2 = cir.load %0 : !cir.ptr<!u32i>, !u32i
// CHECK:   cir.return %2 : !u32i
// CHECK: }

unsigned long long ullfunc() { return 42ull; }
// CHECK: cir.func{{.*}} @_Z7ullfuncv() -> !u64i
// CHECK:   %0 = cir.alloca !u64i, !cir.ptr<!u64i>, ["__retval"] {alignment = 8 : i64}
// CHECK:   %1 = cir.const #cir.int<42> : !u64i
// CHECK:   cir.store %1, %0 : !u64i, !cir.ptr<!u64i>
// CHECK:   %2 = cir.load %0 : !cir.ptr<!u64i>, !u64i
// CHECK:   cir.return %2 : !u64i
// CHECK: }

bool boolfunc() { return true; }
// CHECK: cir.func{{.*}} @_Z8boolfuncv() -> !cir.bool
// CHECK:   %0 = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["__retval"] {alignment = 1 : i64}
// CHECK:   %1 = cir.const #true
// CHECK:   cir.store %1, %0 : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:   %2 = cir.load %0 : !cir.ptr<!cir.bool>, !cir.bool
// CHECK:   cir.return %2 : !cir.bool
// CHECK: }

float floatfunc() { return 42.42f; }
// CHECK: cir.func{{.*}} @_Z9floatfuncv() -> !cir.float
// CHECK:   %0 = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["__retval"] {alignment = 4 : i64}
// CHECK:   %1 = cir.const #cir.fp<4.242
// CHECK:   cir.store %1, %0 : !cir.float, !cir.ptr<!cir.float>
// CHECK:   %2 = cir.load %0 : !cir.ptr<!cir.float>, !cir.float
// CHECK:   cir.return %2 : !cir.float
// CHECK: }

double doublefunc() { return 42.42; }
// CHECK: cir.func{{.*}} @_Z10doublefuncv() -> !cir.double
// CHECK:   %0 = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["__retval"] {alignment = 8 : i64}
// CHECK:   %1 = cir.const #cir.fp<4.242
// CHECK:   cir.store %1, %0 : !cir.double, !cir.ptr<!cir.double>
// CHECK:   %2 = cir.load %0 : !cir.ptr<!cir.double>, !cir.double
// CHECK:   cir.return %2 : !cir.double
// CHECK: }
