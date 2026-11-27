// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,OGCG

// LLVM requires every call in a function that uses constrained FP intrinsics
// to be strictfp, at the call site or on the callee. CIR keeps strictfp on the
// function only. The lowering to the LLVM dialect marks the calls in it,
// including intrinsic calls and the libcalls that LoweringPrepare creates.
//
// ClangIR and classic codegen differ only where classic codegen also adds
// nounwind, which is unrelated to strictfp. The LLVMCIR and OGCG lines cover
// those calls.

void ext(void);
float extf(float);
struct S { int a[32]; };

void plain_direct(void) { ext(); }

// CIR-LABEL: cir.func {{.*}}@plain_direct()
// CIR:         cir.call @ext() : () -> ()

// LLVM-LABEL: define {{.*}}void @plain_direct()
// LLVM-SAME:    #[[PLAIN_FN:[0-9]+]]
// LLVM:         call void @ext(){{$}}

void plain_indirect(void (*fp)(void)) { fp(); }

// LLVM-LABEL: define {{.*}}void @plain_indirect(
// LLVM-SAME:    #[[PLAIN_FN]]
// LLVM:         call void %{{.*}}(){{$}}

void plain_copy(struct S *d, struct S *s) { *d = *s; }

// LLVM-LABEL: define {{.*}}void @plain_copy(
// LLVM-SAME:    #[[PLAIN_FN]]
// LLVM:         call void @llvm.memcpy.p0.p0.i64({{.*}}, i64 128, i1 false){{$}}

void plain_trap(void) { __builtin_trap(); }

// LLVM-LABEL: define {{.*}}void @plain_trap()
// LLVM-SAME:    #[[PLAIN_FN]]
// LLVM:         call void @llvm.trap(){{$}}

#pragma STDC FENV_ACCESS ON

void strict_direct(void) { ext(); }

// The attribute stays on the function in CIR.
// CIR-LABEL: cir.func {{.*}}@strict_direct()
// CIR-SAME:    attributes {{{.*}}strictfp}
// CIR:         cir.call @ext() : () -> ()

// LLVM-LABEL: define {{.*}}void @strict_direct()
// LLVM-SAME:    #[[STRICT_FN:[0-9]+]]
// LLVM:         call void @ext() #[[STRICT_CALL:[0-9]+]]

void strict_indirect(void (*fp)(void)) { fp(); }

// CIR-LABEL: cir.func {{.*}}@strict_indirect(
// CIR-SAME:    attributes {{{.*}}strictfp}
// CIR:         cir.call %{{.*}}() : (!cir.ptr<!cir.func<()>>) -> ()

// LLVM-LABEL: define {{.*}}void @strict_indirect(
// LLVM-SAME:    #[[STRICT_FN]]
// LLVM:         call void %{{.*}}() #[[STRICT_CALL]]

// A call next to a constrained operation.
float strict_mixed(float a) { return extf(a + 1.0f); }

// LLVM-LABEL: define {{.*}}float @strict_mixed(
// LLVM-SAME:    #[[STRICT_FN]]
// LLVM:         call float @llvm.experimental.constrained.fadd.f32({{.*}}) #[[STRICT_CALL]]
// LLVM:         call float @extf(float {{.*}}) #[[STRICT_CALL]]

// An intrinsic that has no constrained variant.
float strict_builtin(float a) { return __builtin_fabsf(a); }

// LLVM-LABEL: define {{.*}}float @strict_builtin(
// LLVM-SAME:    #[[STRICT_FN]]
// LLVM:         call float @llvm.fabs.f32(float %{{.*}}) #[[STRICT_CALL]]

// An intrinsic that the lowering creates for an operation that is not a call.
void strict_copy(struct S *d, struct S *s) { *d = *s; }

// CIR-LABEL: cir.func {{.*}}@strict_copy(
// CIR-SAME:    attributes {{{.*}}strictfp}
// CIR:         cir.copy

// LLVM-LABEL: define {{.*}}void @strict_copy(
// LLVM-SAME:    #[[STRICT_FN]]
// LLVM:         call void @llvm.memcpy.p0.p0.i64({{.*}}, i64 128, i1 false) #[[STRICT_CALL]]

void strict_trap(void) { __builtin_trap(); }

// LLVM-LABEL: define {{.*}}void @strict_trap()
// LLVM-SAME:    #[[STRICT_FN]]
// LLVM:         call void @llvm.trap() #[[STRICT_CALL]]

// A libcall that LoweringPrepare creates.
_Complex float strict_complex_mul(_Complex float a, _Complex float b) {
  return a * b;
}

// LLVM-LABEL: define {{.*}}<2 x float> @strict_complex_mul(
// LLVMCIR:      call <2 x float> @__mulsc3({{.*}}) #[[STRICT_CALL]]
// OGCG:         call <2 x float> @__mulsc3({{.*}}) #[[STRICT_NOUNWIND_CALL:[0-9]+]]

void strict_asm(void) { __asm__ volatile("nop"); }

// CIR-LABEL: cir.func {{.*}}@strict_asm()
// CIR-SAME:    attributes {{{.*}}strictfp}
// CIR:         cir.asm(x86_att,

// LLVM-LABEL: define {{.*}}void @strict_asm()
// LLVM-SAME:    #[[STRICT_FN]]
// LLVMCIR:      call void asm sideeffect "nop", "~{dirflag},~{fpsr},~{flags}"() #[[STRICT_CALL]]
// OGCG:         call void asm sideeffect "nop", "~{dirflag},~{fpsr},~{flags}"() #[[STRICT_NOUNWIND_CALL]]

#pragma STDC FENV_ACCESS OFF

// A call to a strictfp function from a function that is not strictfp.
void plain_calls_strict(void) { strict_direct(); }

// LLVM-LABEL: define {{.*}}void @plain_calls_strict()
// LLVM-SAME:    #[[PLAIN_FN]]
// LLVM:         call void @strict_direct(){{$}}

// LLVM-DAG: attributes #[[STRICT_FN]] = { {{.*}}strictfp
// LLVM-DAG: attributes #[[STRICT_CALL]] = { strictfp }
// OGCG-DAG: attributes #[[STRICT_NOUNWIND_CALL]] = { nounwind strictfp }
