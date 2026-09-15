// RUN: %clang_cc1 -fkeep-persistent-storage-variables -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -fkeep-persistent-storage-variables -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -fkeep-persistent-storage-variables -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

static int g1;
static int g2 = 1;
int g5;
int g6 = 6;

namespace {
  int s4 = 42;
}

struct ST {
  static int s6;
};
int ST::s6 = 7;

// CIR: cir.global "private" appending @llvm.compiler.used = #cir.const_array<[#cir.global_view<@_ZL2g1> : !cir.ptr<!void>, #cir.global_view<@_ZL2g2> : !cir.ptr<!void>, #cir.global_view<@g5> : !cir.ptr<!void>, #cir.global_view<@g6> : !cir.ptr<!void>, #cir.global_view<@_ZN12_GLOBAL__N_12s4E> : !cir.ptr<!void>, #cir.global_view<@_ZN2ST2s6E> : !cir.ptr<!void>]> : !cir.array<!cir.ptr<!void> x 6>
// LLVM: @llvm.compiler.used = appending global [6 x ptr] [ptr @_ZL2g1, ptr @_ZL2g2, ptr @g5, ptr @g6, ptr @_ZN12_GLOBAL__N_12s4E, ptr @_ZN2ST2s6E], section "llvm.metadata"
