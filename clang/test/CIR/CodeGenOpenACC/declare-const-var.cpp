// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

static const int ConstInt = 42;
// CHECK: cir.global "private" constant internal dso_local @_ZL8ConstInt = #cir.int<42> : !s32i

#pragma acc declare copyin(ConstInt)
// CHECK: acc.global_ctor @_ZL8ConstInt_acc_ctor {
// CHECK-NEXT: %[[GET_GLOBAL:.*]] = cir.get_global @_ZL8ConstInt : !cir.ptr<!s32i>
// CHECK-NEXT: %[[COPYIN:.*]] = acc.copyin varPtr(%[[GET_GLOBAL]] : !cir.ptr<!s32i>) name("ConstInt") -> !cir.ptr<!s32i>
// CHECK-NEXT: acc.declare_enter dataOperands(%[[COPYIN]] : !cir.ptr<!s32i>)
