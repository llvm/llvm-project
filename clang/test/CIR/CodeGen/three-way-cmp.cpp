// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck %s --input-file=%t-before.cir --check-prefix=BEFORE
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=AFTER

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=AFTER

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-llvm %s -o %t.ll 2>&1
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm %s -o %t-og.ll 2>&1
// RUN: FileCheck --input-file=%t-og.ll %s -check-prefix=OGCG

#include "../../CodeGenCXX/Inputs/std-compare.h"

// BEFORE: #cmp3way_info_partial_ltn1eq0gt1unn127 = #cir.cmp3way_info<partial, lt = -1, eq = 0, gt = 1, unordered = -127>
// BEFORE: #cmp3way_info_total_ltn1eq0gt1 = #cir.cmp3way_info<total, lt = -1, eq = 0, gt = 1>
// BEFORE: !rec_std3A3A__13A3Apartial_ordering = !cir.record<class "std::__1::partial_ordering" {!s8i}>
// BEFORE: !rec_std3A3A__13A3Astrong_ordering = !cir.record<class "std::__1::strong_ordering" {!s8i}>

auto three_way_total(int x, int y) {
  return x <=> y;
}

// BEFORE: cir.func {{.*}} @_Z15three_way_totalii
// BEFORE:   %{{.+}} = cir.cmp3way #cmp3way_info_total_ltn1eq0gt1 %{{.+}}, %{{.+}} : !s32i -> !s8i
// BEFORE: }

//      AFTER:   cir.func {{.*}} @_Z15three_way_totalii{{.*}}
//      AFTER:   %[[LHS:.*]] = cir.load align(4) %{{.+}} : !cir.ptr<!s32i>, !s32i{{.*}}
// AFTER-NEXT:   %[[RHS:.*]] = cir.load align(4) %{{.+}} : !cir.ptr<!s32i>, !s32i{{.*}}
// AFTER-NEXT:   %[[LT:.*]] = cir.const #cir.int<-1> : !s8i{{.*}}
// AFTER-NEXT:   %[[EQ:.*]] = cir.const #cir.int<0> : !s8i{{.*}}
// AFTER-NEXT:   %[[GT:.*]] = cir.const #cir.int<1> : !s8i{{.*}}
// AFTER-NEXT:   %[[CMP_LT:.*]] = cir.cmp(lt, %[[LHS]], %[[RHS]]) : !s32i, !cir.bool{{.*}}
// AFTER-NEXT:   %[[CMP_EQ:.*]] = cir.cmp(eq, %[[LHS]], %[[RHS]]) : !s32i, !cir.bool{{.*}}
// AFTER-NEXT:   %[[SELECT_1:.*]] = cir.select if %[[CMP_EQ]] then %[[EQ]] else %[[GT]] : (!cir.bool, !s8i, !s8i) -> !s8i{{.*}}
// AFTER-NEXT:   %[[SELECT_2:.*]] = cir.select if %[[CMP_LT]] then %[[LT]] else %[[SELECT_1]] : (!cir.bool, !s8i, !s8i) -> !s8i{{.*}}
// AFTER-NEXT:   %{{.+}} = cir.get_member %{{.+}}[0] {{.*}} "__value_"{{.*}}
// AFTER-NEXT:   cir.store align(1) %[[SELECT_2]], %{{.+}} : !s8i, !cir.ptr<!s8i>{{.*}}
// AFTER-NEXT:   %{{.+}} = cir.load %{{.+}} : !cir.ptr<!rec_std3A3A__13A3Astrong_ordering>, !rec_std3A3A__13A3Astrong_ordering{{.*}}
// AFTER-NEXT:   cir.return %{{.+}} : !rec_std3A3A__13A3Astrong_ordering{{.*}}

// LLVM:  %[[LHS:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-NEXT:  %[[RHS:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-NEXT:  %[[CMP_LT:.*]] = icmp slt i32 %[[LHS]], %[[RHS]]
// LLVM-NEXT:  %[[CMP_EQ:.*]] = icmp eq i32 %[[LHS]], %[[RHS]]
// LLVM-NEXT:  %[[SEL_EQ_GT:.*]] = select i1 %[[CMP_EQ]], i8 0, i8 1
// LLVM-NEXT:  %[[RES:.*]] = select i1 %[[CMP_LT]], i8 -1, i8 %[[SEL_EQ_GT]]

// OGCG:  %[[LHS:.*]] = load i32, ptr %{{.*}}, align 4
// OGCG-NEXT:  %[[RHS:.*]] = load i32, ptr %{{.*}}, align 4
// OGCG-NEXT:  %[[CMP_LT:.*]] = icmp slt i32 %[[LHS]], %[[RHS]]
// OGCG-NEXT:  %[[SEL_EQ_LT:.*]] = select i1 %[[CMP_LT]], i8 -1, i8 1
// OGCG-NEXT:  %[[CMP_EQ:.*]] = icmp eq i32 %[[LHS]], %[[RHS]]
// OGCG-NEXT:  %[[RES:.*]] = select i1 %[[CMP_EQ]], i8 0, i8 %[[SEL_EQ_LT]]

auto three_way_partial(float x, float y) {
  return x <=> y;
}

// BEFORE: cir.func {{.*}} @_Z17three_way_partialff
// BEFORE:   %{{.+}} = cir.cmp3way #cmp3way_info_partial_ltn1eq0gt1unn127 %{{.+}}, %{{.+}} : !cir.float -> !s8i
// BEFORE: }

//      AFTER:   cir.func {{.*}} @_Z17three_way_partialff{{.*}}
//      AFTER:   %[[LHS:.*]] = cir.load align(4) %{{.+}} : !cir.ptr<!cir.float>, !cir.float{{.*}}
// AFTER-NEXT:   %[[RHS:.*]] = cir.load align(4) %{{.+}} : !cir.ptr<!cir.float>, !cir.float{{.*}}
// AFTER-NEXT:   %[[LT:.*]] = cir.const #cir.int<-1> : !s8i{{.*}}
// AFTER-NEXT:   %[[EQ:.*]] = cir.const #cir.int<0> : !s8i{{.*}}
// AFTER-NEXT:   %[[GT:.*]] = cir.const #cir.int<1> : !s8i{{.*}}
// AFTER-NEXT:   %[[CMP_LT:.*]] = cir.cmp(lt, %[[LHS]], %[[RHS]]) : !cir.float, !cir.bool{{.*}}
// AFTER-NEXT:   %[[CMP_EQ:.*]] = cir.cmp(eq, %[[LHS]], %[[RHS]]) : !cir.float, !cir.bool{{.*}}
// AFTER-NEXT:   %[[UNORDERED:.*]] = cir.const #cir.int<-127> : !s8i{{.*}}
// AFTER-NEXT:   %[[CMP_GT:.*]] = cir.cmp(gt, %[[LHS]], %[[RHS]]) : !cir.float, !cir.bool{{.*}}
// AFTER-NEXT:   %[[SELECT_1:.*]] = cir.select if %[[CMP_EQ]] then %[[EQ]] else %[[UNORDERED]] : (!cir.bool, !s8i, !s8i) -> !s8i{{.*}}
// AFTER-NEXT:   %[[SELECT_2:.*]] = cir.select if %[[CMP_GT]] then %[[GT]] else %[[SELECT_1]] : (!cir.bool, !s8i, !s8i) -> !s8i{{.*}}
// AFTER-NEXT:   %[[SELECT_3:.*]] = cir.select if %[[CMP_LT]] then %[[LT]] else %[[SELECT_2]] : (!cir.bool, !s8i, !s8i) -> !s8i{{.*}}
// AFTER-NEXT:   %{{.+}} = cir.get_member %{{.+}}[0] {{.*}} "__value_"{{.*}}
// AFTER-NEXT:   cir.store align(1) %[[SELECT_3]], %{{.+}} : !s8i, !cir.ptr<!s8i>{{.*}}
// AFTER-NEXT:   %{{.+}} = cir.load %{{.+}} : !cir.ptr<!rec_std3A3A__13A3Apartial_ordering>, !rec_std3A3A__13A3Apartial_ordering{{.*}}
// AFTER-NEXT:   cir.return %{{.+}} : !rec_std3A3A__13A3Apartial_ordering{{.*}}

// LLVM:  %[[LHS:.*]] = load float, ptr %{{.*}}, align 4
// LLVM:  %[[RHS:.*]] = load float, ptr %{{.*}}, align 4
// LLVM:  %[[CMP_LT:.*]] = fcmp olt float %[[LHS]], %[[RHS]]
// LLVM:  %[[CMP_EQ:.*]] = fcmp oeq float %[[LHS]], %[[RHS]]
// LLVM:  %[[CMP_GT:.*]] = fcmp ogt float %[[LHS]], %[[RHS]]
// LLVM:  %[[SEL_EQ_UN:.*]] = select i1 %[[CMP_EQ]], i8 0, i8 -127
// LLVM:  %[[SEL_GT_EQUN:.*]] = select i1 %[[CMP_GT]], i8 1, i8 %[[SEL_EQ_UN]]
// LLVM:  %[[RES:.*]] = select i1 %[[CMP_LT]], i8 -1, i8 %[[SEL_GT_EQUN]]

// OGCG:  %[[LHS:.*]] = load float, ptr %{{.*}}, align 4
// OGCG:  %[[RHS:.*]] = load float, ptr %{{.*}}, align 4
// OGCG:  %[[CMP_EQ:.*]] = fcmp oeq float %[[LHS]], %[[RHS]]
// OGCG:  %[[SEL_EQ_UN:.*]] = select i1 %[[CMP_EQ]], i8 0, i8 -127
// OGCG:  %[[CMP_GT:.*]] = fcmp ogt float %[[LHS]], %[[RHS]]
// OGCG:  %[[SEL_GT_EQUN:.*]] = select i1 %[[CMP_GT]], i8 1, i8 %[[SEL_EQ_UN]]
// OGCG:  %[[CMP_LT:.*]] = fcmp olt float %[[LHS]], %[[RHS]]
// OGCG:  %[[RES:.*]] = select i1 %[[CMP_LT]], i8 -1, i8 %[[SEL_GT_EQUN]]
