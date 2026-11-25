// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck %s --input-file=%t-before.cir --check-prefix=BEFORE \
// RUN:   -DFENV="fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>"
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR \
// RUN:   -DFENV="fenv<dynamic_rounding_mode = tonearest, except_mode = masked, strict_except = false>"

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s --input-file=%t-cir.ll --check-prefix=LLVM -DEXCEPT=fpexcept.ignore \
// RUN:   --implicit-check-not="{{= fcmp }}"
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM -DEXCEPT=fpexcept.ignore \
// RUN:   --implicit-check-not="{{= fcmp }}"

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -ffp-exception-behavior=strict -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t-strict.cir 2> %t-strict-before.cir
// RUN: FileCheck %s --input-file=%t-strict-before.cir --check-prefix=BEFORE \
// RUN:   -DFENV="fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>"
// RUN: FileCheck %s --input-file=%t-strict.cir --check-prefix=CIR \
// RUN:   -DFENV="fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>"

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -ffp-exception-behavior=strict -fclangir -emit-llvm %s -o %t-strict-cir.ll
// RUN: FileCheck %s --input-file=%t-strict-cir.ll --check-prefix=LLVM -DEXCEPT=fpexcept.strict \
// RUN:   --implicit-check-not="{{= fcmp }}"
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -ffp-exception-behavior=strict -emit-llvm %s -o %t-strict.ll
// RUN: FileCheck %s --input-file=%t-strict.ll --check-prefix=LLVM -DEXCEPT=fpexcept.strict \
// RUN:   --implicit-check-not="{{= fcmp }}"

// A three-way comparison of floating-point values in a strictfp function. The
// comparisons are quiet constrained intrinsics. The LLVM lines are shared by
// ClangIR and classic codegen, so both emit the same intrinsics with the same
// metadata.
//
// The check lines end before the call-site attributes.
// FIXME: ClangIR does not mark the calls in a strictfp function `strictfp`.
//
// Like classic codegen, CIRGen opens no FP-options scope for the comparison,
// so it takes the floating-point environment in effect around it. In a return
// statement that is the default of the function, FENV and EXCEPT here:
// `fpexcept.ignore` with only the pragma and `fpexcept.strict` with
// -ffp-exception-behavior=strict.

#include "./Inputs/std-compare.h"

#pragma STDC FENV_ACCESS ON

auto three_way_partial(double x, double y) {
  return x <=> y;
}

// BEFORE: cir.func {{.*}} @_Z17three_way_partialdd
// BEFORE:   %{{.+}} = cir.cmp3way #cmpinfo_partial_ltn1eq0gt1unn127 %{{.+}}, %{{.+}} : !cir.double -> !s8i [[FENV]]
// BEFORE: }

// CIR: cir.func {{.*}} @_Z17three_way_partialdd
// CIR:   %{{.+}} = cir.fcmp oeq %[[LHS:.*]], %[[RHS:.*]] : !cir.double [[FENV]]
// CIR:   %{{.+}} = cir.fcmp ogt %[[LHS]], %[[RHS]] : !cir.double [[FENV]]
// CIR:   %{{.+}} = cir.fcmp olt %[[LHS]], %[[RHS]] : !cir.double [[FENV]]

// LLVM-LABEL: define{{.*}} i8 @_Z17three_way_partialdd(
// LLVM:   %[[LHS:.*]] = load double, ptr %{{.*}}, align 8
// LLVM:   %[[RHS:.*]] = load double, ptr %{{.*}}, align 8
// LLVM:   %[[CMP_EQ:.*]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %[[LHS]], double %[[RHS]], metadata !"oeq", metadata !"[[EXCEPT]]")
// LLVM:   %[[SEL_EQ_UN:.*]] = select i1 %[[CMP_EQ]], i8 0, i8 -127
// LLVM:   %[[CMP_GT:.*]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %[[LHS]], double %[[RHS]], metadata !"ogt", metadata !"[[EXCEPT]]")
// LLVM:   %[[SEL_GT_EQUN:.*]] = select i1 %[[CMP_GT]], i8 1, i8 %[[SEL_EQ_UN]]
// LLVM:   %[[CMP_LT:.*]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %[[LHS]], double %[[RHS]], metadata !"olt", metadata !"[[EXCEPT]]")
// LLVM:   %{{.*}} = select i1 %[[CMP_LT]], i8 -1, i8 %[[SEL_GT_EQUN]]

// The addition has the floating-point environment of its expression, the
// comparison the default of the function.

auto three_way_partial_of_sum(double x, double y, double z) {
  return (x + y) <=> z;
}

// BEFORE: cir.func {{.*}} @_Z24three_way_partial_of_sumddd
// BEFORE:   %[[SUM:.*]] = cir.fadd %{{.+}}, %{{.+}} : !cir.double fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>
// BEFORE:   %{{.+}} = cir.cmp3way #cmpinfo_partial_ltn1eq0gt1unn127 %[[SUM]], %{{.+}} : !cir.double -> !s8i [[FENV]]
// BEFORE: }

// LLVM-LABEL: define{{.*}} i8 @_Z24three_way_partial_of_sumddd(
// LLVM:   %[[SUM:.*]] = call double @llvm.experimental.constrained.fadd.f64(double %{{.*}}, double %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// LLVM:   %[[RHS:.*]] = load double, ptr %{{.*}}, align 8
// LLVM:   %{{.*}} = call i1 @llvm.experimental.constrained.fcmp.f64(double %[[SUM]], double %[[RHS]], metadata !"oeq", metadata !"[[EXCEPT]]")
// LLVM:   %{{.*}} = call i1 @llvm.experimental.constrained.fcmp.f64(double %[[SUM]], double %[[RHS]], metadata !"ogt", metadata !"[[EXCEPT]]")
// LLVM:   %{{.*}} = call i1 @llvm.experimental.constrained.fcmp.f64(double %[[SUM]], double %[[RHS]], metadata !"olt", metadata !"[[EXCEPT]]")

// A call expression opens an FP-options scope, here for the call of
// `operator<`. A comparison in its arguments has the floating-point
// environment of the pragma.

bool three_way_partial_less(double x, double y) {
  return (x <=> y) < 0;
}

// BEFORE: cir.func {{.*}} @_Z22three_way_partial_lessdd
// BEFORE:   %{{.+}} = cir.cmp3way #cmpinfo_partial_ltn1eq0gt1unn127 %{{.+}}, %{{.+}} : !cir.double -> !s8i fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>
// BEFORE: }

// LLVM-LABEL: define{{.*}} i1 @_Z22three_way_partial_lessdd(
// LLVM:   %[[LHS:.*]] = load double, ptr %{{.*}}, align 8
// LLVM:   %[[RHS:.*]] = load double, ptr %{{.*}}, align 8
// LLVM:   %{{.*}} = call i1 @llvm.experimental.constrained.fcmp.f64(double %[[LHS]], double %[[RHS]], metadata !"oeq", metadata !"fpexcept.strict")
// LLVM:   %{{.*}} = call i1 @llvm.experimental.constrained.fcmp.f64(double %[[LHS]], double %[[RHS]], metadata !"ogt", metadata !"fpexcept.strict")
// LLVM:   %{{.*}} = call i1 @llvm.experimental.constrained.fcmp.f64(double %[[LHS]], double %[[RHS]], metadata !"olt", metadata !"fpexcept.strict")
// LLVM:   call {{.*}}i1 @_ZNSt3__1ltENS_16partial_orderingEMNS_19_CmpUnspecifiedTypeEFvvE(

// An integer three-way comparison has no floating-point environment.

auto three_way_strong(int x, int y) {
  return x <=> y;
}

// BEFORE: cir.func {{.*}} @_Z16three_way_strongii
// BEFORE:   %{{.+}} = cir.cmp3way #cmpinfo_strong_ltn1eq0gt1 %{{.+}}, %{{.+}} : !s32i -> !s8i{{$}}
// BEFORE: }

// CIR: cir.func {{.*}} @_Z16three_way_strongii
// CIR:   %[[CMP_LT:.*]] = cir.cmp lt %[[LHS:.*]], %[[RHS:.*]] : !s32i{{ loc|$}}
// CIR:   %[[CMP_EQ:.*]] = cir.cmp eq %[[LHS]], %[[RHS]] : !s32i{{ loc|$}}

// LLVM-LABEL: define{{.*}} i8 @_Z16three_way_strongii(
// LLVM:   %[[CMP_LT:.*]] = icmp slt i32 %[[LHS:.*]], %[[RHS:.*]]
// LLVM:   %[[SEL_LT_GT:.*]] = select i1 %[[CMP_LT]], i8 -1, i8 1
// LLVM:   %[[CMP_EQ:.*]] = icmp eq i32 %[[LHS]], %[[RHS]]
// LLVM:   %{{.*}} = select i1 %[[CMP_EQ]], i8 0, i8 %[[SEL_LT_GT]]
