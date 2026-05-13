// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

  constexpr const int &normal = 42;
// CIR: cir.global "private" constant external @_ZGR6normal_ = #cir.int<42> : !s32i
// CIR: cir.global constant external @normal = #cir.global_view<@_ZGR6normal_> : !cir.ptr<!s32i>
// LLVM: @_ZGR6normal_ = {{.*}}constant i32 42, align 4
// LLVM: @normal = constant ptr @_ZGR6normal_, align 8

struct SelfRef {
  int *p = ints;
  int ints[3] = {1, 2, 3};
};
constexpr const SelfRef &sr = SelfRef();
// CIR: cir.global "private" constant external @_ZGR2sr_ = #cir.const_record<{#cir.global_view<@_ZGR2sr_, [1 : i32]> : !cir.ptr<!s32i>, #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i]> : !cir.array<!s32i x 3>}> 
// CIR: cir.global constant external @sr = #cir.global_view<@_ZGR2sr_> : !cir.ptr<!rec_SelfRef>
// LLVM: @_ZGR2sr_ = {{.*}}constant { ptr, [3 x i32] } { ptr getelementptr {{.*}}(i8, ptr @_ZGR2sr_, i64 8), [3 x i32] [i32 1, i32 2, i32 3] }, align 8
// LLVM: @sr = constant ptr @_ZGR2sr_, align 8

struct MultiSelfRef {
  int *p = ints;
  int *q = ints;
  int ints[3] = {1, 2, 3};
};

constexpr const MultiSelfRef &msr = MultiSelfRef();
// CIR: cir.global "private" constant external @_ZGR3msr_ = #cir.const_record<{#cir.global_view<@_ZGR3msr_, [2 : i32]> : !cir.ptr<!s32i>, #cir.global_view<@_ZGR3msr_, [2 : i32]> : !cir.ptr<!s32i>, #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i]> : !cir.array<!s32i x 3>}>
// CIR: cir.global constant external @msr = #cir.global_view<@_ZGR3msr_> : !cir.ptr<!rec_MultiSelfRef>
// LLVM: @_ZGR3msr_ = {{.*}}constant { ptr, ptr, [3 x i32] } { ptr getelementptr {{.*}}(i8, ptr @_ZGR3msr_, i64 16), ptr getelementptr {{.*}}(i8, ptr @_ZGR3msr_, i64 16), [3 x i32] [i32 1, i32 2, i32 3] }, align 8
// LLVM: @msr = constant ptr @_ZGR3msr_, align 8

