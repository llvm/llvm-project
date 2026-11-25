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
// CHECK: cir.func{{.*}} @_Z7intfuncv() -> (!s32i {llvm.noundef})
// CHECK:   %0 = cir.alloca "" align(4) return_slot : !cir.ptr<!s32i>
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
// CHECK: cir.func{{.*}} @_Z6scopesv() -> (!s32i {llvm.noundef})
// CHECK:   %0 = cir.alloca "" align(4) return_slot : !cir.ptr<!s32i>
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
// CHECK: cir.func{{.*}} @_Z8longfuncv() -> (!s64i {llvm.noundef})
// CHECK:   %0 = cir.alloca "" align(8) return_slot : !cir.ptr<!s64i>
// CHECK:   %1 = cir.const #cir.int<42> : !s64i
// CHECK:   cir.store %1, %0 : !s64i, !cir.ptr<!s64i>
// CHECK:   %2 = cir.load %0 : !cir.ptr<!s64i>, !s64i
// CHECK:   cir.return %2 : !s64i
// CHECK: }

unsigned unsignedfunc() { return 42u; }
// CHECK: cir.func{{.*}} @_Z12unsignedfuncv() -> (!u32i {llvm.noundef})
// CHECK:   %0 = cir.alloca "" align(4) return_slot : !cir.ptr<!u32i>
// CHECK:   %1 = cir.const #cir.int<42> : !u32i
// CHECK:   cir.store %1, %0 : !u32i, !cir.ptr<!u32i>
// CHECK:   %2 = cir.load %0 : !cir.ptr<!u32i>, !u32i
// CHECK:   cir.return %2 : !u32i
// CHECK: }

unsigned long long ullfunc() { return 42ull; }
// CHECK: cir.func{{.*}} @_Z7ullfuncv() -> (!u64i {llvm.noundef})
// CHECK:   %0 = cir.alloca "" align(8) return_slot : !cir.ptr<!u64i>
// CHECK:   %1 = cir.const #cir.int<42> : !u64i
// CHECK:   cir.store %1, %0 : !u64i, !cir.ptr<!u64i>
// CHECK:   %2 = cir.load %0 : !cir.ptr<!u64i>, !u64i
// CHECK:   cir.return %2 : !u64i
// CHECK: }

bool boolfunc() { return true; }
// CHECK: cir.func{{.*}} @_Z8boolfuncv() -> (!cir.bool {llvm.noundef, llvm.zeroext})
// CHECK:   %0 = cir.alloca "" align(1) return_slot : !cir.ptr<!cir.bool>
// CHECK:   %1 = cir.const #true
// CHECK:   cir.store %1, %0 : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:   %2 = cir.load %0 : !cir.ptr<!cir.bool>, !cir.bool
// CHECK:   cir.return %2 : !cir.bool
// CHECK: }

float floatfunc() { return 42.42f; }
// CHECK: cir.func{{.*}} @_Z9floatfuncv() -> (!cir.float {llvm.noundef})
// CHECK:   %0 = cir.alloca "" align(4) return_slot : !cir.ptr<!cir.float>
// CHECK:   %1 = cir.const #cir.fp<4.242
// CHECK:   cir.store %1, %0 : !cir.float, !cir.ptr<!cir.float>
// CHECK:   %2 = cir.load %0 : !cir.ptr<!cir.float>, !cir.float
// CHECK:   cir.return %2 : !cir.float
// CHECK: }

double doublefunc() { return 42.42; }
// CHECK: cir.func{{.*}} @_Z10doublefuncv() -> (!cir.double {llvm.noundef})
// CHECK:   %0 = cir.alloca "" align(8) return_slot : !cir.ptr<!cir.double>
// CHECK:   %1 = cir.const #cir.fp<4.242
// CHECK:   cir.store %1, %0 : !cir.double, !cir.ptr<!cir.double>
// CHECK:   %2 = cir.load %0 : !cir.ptr<!cir.double>, !cir.double
// CHECK:   cir.return %2 : !cir.double
// CHECK: }

int user_retval() {
  int __retval = 1;
  return __retval;
}
// CHECK: cir.func{{.*}} @_Z11user_retvalv() -> (!s32i {llvm.noundef})
// CHECK:   %0 = cir.alloca "" align(4) return_slot : !cir.ptr<!s32i>
// CHECK:   %1 = cir.alloca "__retval" align(4) init : !cir.ptr<!s32i>
