// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=BEFORE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=AFTER

#include "Inputs/std-compare.h"

// BEFORE: #cmp3way_info_partial_ltn1eq0gt1unn127_ = #cir.cmp3way_info<partial, lt = -1, eq = 0, gt = 1, unordered = -127>
// BEFORE: #cmp3way_info_strong_ltn1eq0gt1_ = #cir.cmp3way_info<strong, lt = -1, eq = 0, gt = 1>
// BEFORE: !ty_22std3A3A__13A3Apartial_ordering22 = !cir.struct<class "std::__1::partial_ordering" {!cir.int<s, 8>}
// BEFORE: !ty_22std3A3A__13A3Astrong_ordering22 = !cir.struct<class "std::__1::strong_ordering" {!cir.int<s, 8>}

auto three_way_strong(int x, int y) {
  return x <=> y;
}

// BEFORE: cir.func @_Z16three_way_strongii
// BEFORE:   %{{.+}} = cir.cmp3way(%{{.+}} : !s32i, %{{.+}}, #cmp3way_info_strong_ltn1eq0gt1_) : !s8i
// BEFORE: }

//      AFTER: cir.func @_Z16three_way_strongii
//      AFTER:   %[[#LHS:]] = cir.load %{{.+}} : cir.ptr <!s32i>, !s32i
// AFTER-NEXT:   %[[#RHS:]] = cir.load %{{.+}} : cir.ptr <!s32i>, !s32i
// AFTER-NEXT:   %[[#LT:]] = cir.const(#cir.int<-1> : !s8i) : !s8i
// AFTER-NEXT:   %[[#EQ:]] = cir.const(#cir.int<0> : !s8i) : !s8i
// AFTER-NEXT:   %[[#GT:]] = cir.const(#cir.int<1> : !s8i) : !s8i
// AFTER-NEXT:   %[[#CMP_LT:]] = cir.cmp(lt, %[[#LHS]], %[[#RHS]]) : !s32i, !cir.bool
// AFTER-NEXT:   %[[#CMP_EQ:]] = cir.cmp(eq, %[[#LHS]], %[[#RHS]]) : !s32i, !cir.bool
// AFTER-NEXT:   %[[#CMP_EQ_RES:]] = cir.ternary(%[[#CMP_EQ]], true {
// AFTER-NEXT:     cir.yield %[[#EQ]] : !s8i
// AFTER-NEXT:   }, false {
// AFTER-NEXT:     cir.yield %[[#GT]] : !s8i
// AFTER-NEXT:   }) : (!cir.bool) -> !s8i
// AFTER-NEXT:   %{{.+}} = cir.ternary(%[[#CMP_LT]], true {
// AFTER-NEXT:     cir.yield %[[#LT]] : !s8i
// AFTER-NEXT:   }, false {
// AFTER-NEXT:     cir.yield %[[#CMP_EQ_RES]] : !s8i
// AFTER-NEXT:   }) : (!cir.bool) -> !s8i
//      AFTER: }

auto three_way_weak(float x, float y) {
  return x <=> y;
}

// BEFORE: cir.func @_Z14three_way_weakff
// BEFORE:   %{{.+}} = cir.cmp3way(%{{.+}} : !cir.float, %{{.+}}, #cmp3way_info_partial_ltn1eq0gt1unn127_) : !s8i
// BEFORE: }

//      AFTER: cir.func @_Z14three_way_weakff
//      AFTER:   %[[#LHS:]] = cir.load %0 : cir.ptr <!cir.float>, !cir.float
// AFTER-NEXT:   %[[#RHS:]] = cir.load %1 : cir.ptr <!cir.float>, !cir.float
// AFTER-NEXT:   %[[#LT:]] = cir.const(#cir.int<-1> : !s8i) : !s8i
// AFTER-NEXT:   %[[#EQ:]] = cir.const(#cir.int<0> : !s8i) : !s8i
// AFTER-NEXT:   %[[#GT:]] = cir.const(#cir.int<1> : !s8i) : !s8i
// AFTER-NEXT:   %[[#UNORDERED:]] = cir.const(#cir.int<-127> : !s8i) : !s8i
// AFTER-NEXT:   %[[#CMP_LT:]] = cir.cmp(lt, %[[#LHS]], %[[#RHS]]) : !cir.float, !cir.bool
// AFTER-NEXT:   %[[#CMP_EQ:]] = cir.cmp(eq, %[[#LHS]], %[[#RHS]]) : !cir.float, !cir.bool
// AFTER-NEXT:   %[[#CMP_GT:]] = cir.cmp(gt, %[[#LHS]], %[[#RHS]]) : !cir.float, !cir.bool
// AFTER-NEXT:   %[[#CMP_EQ_RES:]] = cir.ternary(%[[#CMP_EQ]], true {
// AFTER-NEXT:     cir.yield %[[#EQ]] : !s8i
// AFTER-NEXT:   }, false {
// AFTER-NEXT:     cir.yield %[[#UNORDERED]] : !s8i
// AFTER-NEXT:   }) : (!cir.bool) -> !s8i
// AFTER-NEXT:   %[[#CMP_GT_RES:]] = cir.ternary(%[[#CMP_GT]], true {
// AFTER-NEXT:     cir.yield %[[#GT]] : !s8i
// AFTER-NEXT:   }, false {
// AFTER-NEXT:     cir.yield %[[#CMP_EQ_RES]] : !s8i
// AFTER-NEXT:   }) : (!cir.bool) -> !s8i
// AFTER-NEXT:   %{{.+}} = cir.ternary(%[[#CMP_LT]], true {
// AFTER-NEXT:     cir.yield %[[#LT]] : !s8i
// AFTER-NEXT:   }, false {
// AFTER-NEXT:     cir.yield %[[#CMP_GT_RES]] : !s8i
// AFTER-NEXT:   }) : (!cir.bool) -> !s8i
//      AFTER: }
