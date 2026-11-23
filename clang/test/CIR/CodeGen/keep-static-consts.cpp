// RUN: %clang_cc1 -fkeep-static-consts -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -fkeep-static-consts -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -fkeep-static-consts -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

static const char srcvers[] = "xyz";
extern const int b = 1;
const char srcvers2[] = "abc";
constexpr int N = 2;

// CIR: cir.global "private" appending @llvm.compiler.used = #cir.const_array<[#cir.global_view<@_ZL7srcvers> : !cir.ptr<!void>, #cir.global_view<@b> : !cir.ptr<!void>, #cir.global_view<@_ZL8srcvers2> : !cir.ptr<!void>, #cir.global_view<@_ZL1N> : !cir.ptr<!void>]> : !cir.array<!cir.ptr<!void> x 4>
// LLVM: @llvm.compiler.used = appending global [4 x ptr] [ptr @_ZL7srcvers, ptr @b, ptr @_ZL8srcvers2, ptr @_ZL1N], section "llvm.metadata"
