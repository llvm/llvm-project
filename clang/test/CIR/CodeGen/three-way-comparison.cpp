// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=BEFORE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=AFTER
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -DNON_CANONICAL_CMP_RESULTS -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=NONCANONICAL-BEFORE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -DNON_CANONICAL_CMP_RESULTS -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=NONCANONICAL-AFTER

#include "Inputs/std-compare.h"

// BEFORE: #cmp3way_info_partial_ltn1eq0gt1unn127_ = #cir.cmp3way_info<partial, lt = -1, eq = 0, gt = 1, unordered = -127>
// BEFORE: #cmp3way_info_strong_ltn1eq0gt1_ = #cir.cmp3way_info<strong, lt = -1, eq = 0, gt = 1>
// BEFORE: !ty_std3A3A__13A3Apartial_ordering = !cir.struct<class "std::__1::partial_ordering" {!cir.int<s, 8>}
// BEFORE: !ty_std3A3A__13A3Astrong_ordering = !cir.struct<class "std::__1::strong_ordering" {!cir.int<s, 8>}

auto three_way_strong(int x, int y) {
  return x <=> y;
}

// BEFORE: cir.func @_Z16three_way_strongii
// BEFORE:   %{{.+}} = cir.cmp3way(%{{.+}} : !s32i, %{{.+}}, #cmp3way_info_strong_ltn1eq0gt1_) : !s8i
// BEFORE: }

// AFTER: cir.func @_Z16three_way_strongii
// AFTER:   %{{.+}} = cir.cmp3way(%{{.+}} : !s32i, %{{.+}}, #cmp3way_info_strong_ltn1eq0gt1_) : !s8i
// AFTER: }

// NONCANONICAL-BEFORE: #cmp3way_info_strong_lt1eq2gt3_ = #cir.cmp3way_info<strong, lt = 1, eq = 2, gt = 3>
// NONCANONICAL-BEFORE: cir.func @_Z16three_way_strongii
// NONCANONICAL-BEFORE:   %{{.+}} = cir.cmp3way(%{{.+}} : !s32i, %{{.+}}, #cmp3way_info_strong_lt1eq2gt3_) : !s8i
// NONCANONICAL-BEFORE: }

//      NONCANONICAL-AFTER: #cmp3way_info_strong_ltn1eq0gt1_ = #cir.cmp3way_info<strong, lt = -1, eq = 0, gt = 1>
//      NONCANONICAL-AFTER: cir.func @_Z16three_way_strongii
//      NONCANONICAL-AFTER:   %[[#CMP3WAY_RESULT:]] = cir.cmp3way(%{{.+}} : !s32i, %{{.+}}, #cmp3way_info_strong_ltn1eq0gt1_) : !s8i
// NONCANONICAL-AFTER-NEXT:   %[[#NEGONE:]] = cir.const #cir.int<-1> : !s8i
// NONCANONICAL-AFTER-NEXT:   %[[#ONE:]] = cir.const #cir.int<1> : !s8i
// NONCANONICAL-AFTER-NEXT:   %[[#CMP_TO_NEGONE:]] = cir.cmp(eq, %[[#CMP3WAY_RESULT]], %[[#NEGONE]]) : !s8i, !cir.bool
// NONCANONICAL-AFTER-NEXT:   %[[#A:]] = cir.select if %[[#CMP_TO_NEGONE]] then %[[#ONE]] else %[[#CMP3WAY_RESULT]] : (!cir.bool, !s8i, !s8i) -> !s8i
// NONCANONICAL-AFTER-NEXT:   %[[#ZERO:]] = cir.const #cir.int<0> : !s8i
// NONCANONICAL-AFTER-NEXT:   %[[#TWO:]] = cir.const #cir.int<2> : !s8i
// NONCANONICAL-AFTER-NEXT:   %[[#CMP_TO_ZERO:]] = cir.cmp(eq, %[[#A]], %[[#ZERO]]) : !s8i, !cir.bool
// NONCANONICAL-AFTER-NEXT:   %[[#B:]] = cir.select if %[[#CMP_TO_ZERO]] then %[[#TWO]] else %[[#A]] : (!cir.bool, !s8i, !s8i) -> !s8i
// NONCANONICAL-AFTER-NEXT:   %[[#ONE2:]] = cir.const #cir.int<1> : !s8i
// NONCANONICAL-AFTER-NEXT:   %[[#THREE:]] = cir.const #cir.int<3> : !s8i
// NONCANONICAL-AFTER-NEXT:   %[[#CMP_TO_ONE:]] = cir.cmp(eq, %[[#B]], %[[#ONE2]]) : !s8i, !cir.bool
// NONCANONICAL-AFTER-NEXT:   %{{.+}} = cir.select if %[[#CMP_TO_ONE]] then %[[#THREE]] else %[[#B]] : (!cir.bool, !s8i, !s8i) -> !s8i
//      NONCANONICAL-AFTER: }

auto three_way_weak(float x, float y) {
  return x <=> y;
}

// BEFORE: cir.func @_Z14three_way_weakff
// BEFORE:   %{{.+}} = cir.cmp3way(%{{.+}} : !cir.float, %{{.+}}, #cmp3way_info_partial_ltn1eq0gt1unn127_) : !s8i
// BEFORE: }

//      AFTER: cir.func @_Z14three_way_weakff
//      AFTER:   %[[#LHS:]] = cir.load %0 : !cir.ptr<!cir.float>, !cir.float
// AFTER-NEXT:   %[[#RHS:]] = cir.load %1 : !cir.ptr<!cir.float>, !cir.float
// AFTER-NEXT:   %[[#LT:]] = cir.const #cir.int<-1> : !s8i
// AFTER-NEXT:   %[[#EQ:]] = cir.const #cir.int<0> : !s8i
// AFTER-NEXT:   %[[#GT:]] = cir.const #cir.int<1> : !s8i
// AFTER-NEXT:   %[[#UNORDERED:]] = cir.const #cir.int<-127> : !s8i
// AFTER-NEXT:   %[[#CMP_LT:]] = cir.cmp(lt, %[[#LHS]], %[[#RHS]]) : !cir.float, !cir.bool
// AFTER-NEXT:   %[[#CMP_EQ:]] = cir.cmp(eq, %[[#LHS]], %[[#RHS]]) : !cir.float, !cir.bool
// AFTER-NEXT:   %[[#CMP_GT:]] = cir.cmp(gt, %[[#LHS]], %[[#RHS]]) : !cir.float, !cir.bool
// AFTER-NEXT:   %[[#CMP_EQ_RES:]] = cir.select if %[[#CMP_EQ]] then %[[#EQ]] else %[[#UNORDERED]] : (!cir.bool, !s8i, !s8i) -> !s8i
// AFTER-NEXT:   %[[#CMP_GT_RES:]] = cir.select if %[[#CMP_GT]] then %[[#GT]] else %[[#CMP_EQ_RES]] : (!cir.bool, !s8i, !s8i) -> !s8i
// AFTER-NEXT:   %{{.+}} = cir.select if %[[#CMP_LT]] then %[[#LT]] else %[[#CMP_GT_RES]] : (!cir.bool, !s8i, !s8i) -> !s8i
//      AFTER: }
