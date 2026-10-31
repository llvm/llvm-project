// Check FPOptions on scalar and vector arithmetic, both in CIR and after
// lowering. Classic CodeGen uses the same LLVM checks.
// Contraction is disabled unless explicitly enabled so each flag is isolated.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR,CIR-NONE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-NONE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffp-contract=off -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-NONE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -menable-no-nans -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR,CIR-NNAN
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -menable-no-nans -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-NNAN
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffp-contract=off -menable-no-nans -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-NNAN
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -menable-no-infs -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR,CIR-NINF
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -menable-no-infs -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-NINF
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffp-contract=off -menable-no-infs -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-NINF
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -fno-signed-zeros -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR,CIR-NSZ
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -fno-signed-zeros -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-NSZ
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffp-contract=off -fno-signed-zeros -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-NSZ
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -freciprocal-math -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR,CIR-ARCP
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -freciprocal-math -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-ARCP
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffp-contract=off -freciprocal-math -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-ARCP
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -fapprox-func -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR,CIR-AFN
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -fapprox-func -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-AFN
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffp-contract=off -fapprox-func -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-AFN
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -mreassociate -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR,CIR-REASSOC
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -mreassociate -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-REASSOC
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffp-contract=off -mreassociate -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-REASSOC
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -menable-no-nans -menable-no-infs -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR,CIR-FINITE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -menable-no-nans -menable-no-infs -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-FINITE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffp-contract=off -menable-no-nans -menable-no-infs -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-FINITE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -funsafe-math-optimizations -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR,CIR-UNSAFE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -funsafe-math-optimizations -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-UNSAFE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffp-contract=off -funsafe-math-optimizations -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-UNSAFE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -ffast-math -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR,CIR-FAST-NO-CONTRACT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -ffast-math -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-FAST-NO-CONTRACT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffp-contract=off -ffast-math -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-FAST-NO-CONTRACT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -ffast-math -ffp-contract=fast -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR,CIR-FAST
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -ffast-math -ffp-contract=fast -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-FAST
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffp-contract=off -ffast-math -ffp-contract=fast -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-FAST

float scalar(float a, float b, float c) {
  return ((a + b) * (a - b)) / c;
}
// CIR-LABEL: cir.func {{.*}}@scalar(
// LLVM-LABEL: define {{.*}}@scalar(
// CIR-NONE: cir.fadd %{{.*}}, %{{.*}} : !cir.float{{( loc.*)?$}}
// CIR-NONE: cir.fsub %{{.*}}, %{{.*}} : !cir.float{{( loc.*)?$}}
// CIR-NONE: cir.fmul %{{.*}}, %{{.*}} : !cir.float{{( loc.*)?$}}
// CIR-NONE: cir.fdiv %{{.*}}, %{{.*}} : !cir.float{{( loc.*)?$}}
// LLVM-NONE: fadd float %{{.*}}, %{{.*}}
// LLVM-NONE: fsub float %{{.*}}, %{{.*}}
// LLVM-NONE: fmul float %{{.*}}, %{{.*}}
// LLVM-NONE: fdiv float %{{.*}}, %{{.*}}
// CIR-NNAN: cir.fadd %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [nnan]>{{( loc.*)?$}}
// CIR-NNAN: cir.fsub %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [nnan]>{{( loc.*)?$}}
// CIR-NNAN: cir.fmul %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [nnan]>{{( loc.*)?$}}
// CIR-NNAN: cir.fdiv %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [nnan]>{{( loc.*)?$}}
// LLVM-NNAN: fadd nnan float %{{.*}}, %{{.*}}
// LLVM-NNAN: fsub nnan float %{{.*}}, %{{.*}}
// LLVM-NNAN: fmul nnan float %{{.*}}, %{{.*}}
// LLVM-NNAN: fdiv nnan float %{{.*}}, %{{.*}}
// CIR-NINF: cir.fadd %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [ninf]>{{( loc.*)?$}}
// CIR-NINF: cir.fsub %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [ninf]>{{( loc.*)?$}}
// CIR-NINF: cir.fmul %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [ninf]>{{( loc.*)?$}}
// CIR-NINF: cir.fdiv %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [ninf]>{{( loc.*)?$}}
// LLVM-NINF: fadd ninf float %{{.*}}, %{{.*}}
// LLVM-NINF: fsub ninf float %{{.*}}, %{{.*}}
// LLVM-NINF: fmul ninf float %{{.*}}, %{{.*}}
// LLVM-NINF: fdiv ninf float %{{.*}}, %{{.*}}
// CIR-NSZ: cir.fadd %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [nsz]>{{( loc.*)?$}}
// CIR-NSZ: cir.fsub %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [nsz]>{{( loc.*)?$}}
// CIR-NSZ: cir.fmul %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [nsz]>{{( loc.*)?$}}
// CIR-NSZ: cir.fdiv %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [nsz]>{{( loc.*)?$}}
// LLVM-NSZ: fadd nsz float %{{.*}}, %{{.*}}
// LLVM-NSZ: fsub nsz float %{{.*}}, %{{.*}}
// LLVM-NSZ: fmul nsz float %{{.*}}, %{{.*}}
// LLVM-NSZ: fdiv nsz float %{{.*}}, %{{.*}}
// CIR-ARCP: cir.fadd %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [arcp]>{{( loc.*)?$}}
// CIR-ARCP: cir.fsub %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [arcp]>{{( loc.*)?$}}
// CIR-ARCP: cir.fmul %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [arcp]>{{( loc.*)?$}}
// CIR-ARCP: cir.fdiv %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [arcp]>{{( loc.*)?$}}
// LLVM-ARCP: fadd arcp float %{{.*}}, %{{.*}}
// LLVM-ARCP: fsub arcp float %{{.*}}, %{{.*}}
// LLVM-ARCP: fmul arcp float %{{.*}}, %{{.*}}
// LLVM-ARCP: fdiv arcp float %{{.*}}, %{{.*}}
// CIR-AFN: cir.fadd %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [afn]>{{( loc.*)?$}}
// CIR-AFN: cir.fsub %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [afn]>{{( loc.*)?$}}
// CIR-AFN: cir.fmul %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [afn]>{{( loc.*)?$}}
// CIR-AFN: cir.fdiv %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [afn]>{{( loc.*)?$}}
// LLVM-AFN: fadd afn float %{{.*}}, %{{.*}}
// LLVM-AFN: fsub afn float %{{.*}}, %{{.*}}
// LLVM-AFN: fmul afn float %{{.*}}, %{{.*}}
// LLVM-AFN: fdiv afn float %{{.*}}, %{{.*}}
// CIR-REASSOC: cir.fadd %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [reassoc]>{{( loc.*)?$}}
// CIR-REASSOC: cir.fsub %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [reassoc]>{{( loc.*)?$}}
// CIR-REASSOC: cir.fmul %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [reassoc]>{{( loc.*)?$}}
// CIR-REASSOC: cir.fdiv %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [reassoc]>{{( loc.*)?$}}
// LLVM-REASSOC: fadd reassoc float %{{.*}}, %{{.*}}
// LLVM-REASSOC: fsub reassoc float %{{.*}}, %{{.*}}
// LLVM-REASSOC: fmul reassoc float %{{.*}}, %{{.*}}
// LLVM-REASSOC: fdiv reassoc float %{{.*}}, %{{.*}}
// CIR-FINITE: cir.fadd %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [nnan, ninf]>{{( loc.*)?$}}
// CIR-FINITE: cir.fsub %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [nnan, ninf]>{{( loc.*)?$}}
// CIR-FINITE: cir.fmul %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [nnan, ninf]>{{( loc.*)?$}}
// CIR-FINITE: cir.fdiv %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [nnan, ninf]>{{( loc.*)?$}}
// LLVM-FINITE: fadd nnan ninf float %{{.*}}, %{{.*}}
// LLVM-FINITE: fsub nnan ninf float %{{.*}}, %{{.*}}
// LLVM-FINITE: fmul nnan ninf float %{{.*}}, %{{.*}}
// LLVM-FINITE: fdiv nnan ninf float %{{.*}}, %{{.*}}
// CIR-UNSAFE: cir.fadd %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [nsz, arcp, afn, reassoc]>{{( loc.*)?$}}
// CIR-UNSAFE: cir.fsub %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [nsz, arcp, afn, reassoc]>{{( loc.*)?$}}
// CIR-UNSAFE: cir.fmul %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [nsz, arcp, afn, reassoc]>{{( loc.*)?$}}
// CIR-UNSAFE: cir.fdiv %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [nsz, arcp, afn, reassoc]>{{( loc.*)?$}}
// LLVM-UNSAFE: fadd reassoc nsz arcp afn float %{{.*}}, %{{.*}}
// LLVM-UNSAFE: fsub reassoc nsz arcp afn float %{{.*}}, %{{.*}}
// LLVM-UNSAFE: fmul reassoc nsz arcp afn float %{{.*}}, %{{.*}}
// LLVM-UNSAFE: fdiv reassoc nsz arcp afn float %{{.*}}, %{{.*}}
// CIR-FAST-NO-CONTRACT: cir.fadd %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [nnan, ninf, nsz, arcp, afn, reassoc]>{{( loc.*)?$}}
// CIR-FAST-NO-CONTRACT: cir.fsub %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [nnan, ninf, nsz, arcp, afn, reassoc]>{{( loc.*)?$}}
// CIR-FAST-NO-CONTRACT: cir.fmul %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [nnan, ninf, nsz, arcp, afn, reassoc]>{{( loc.*)?$}}
// CIR-FAST-NO-CONTRACT: cir.fdiv %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [nnan, ninf, nsz, arcp, afn, reassoc]>{{( loc.*)?$}}
// LLVM-FAST-NO-CONTRACT: fadd reassoc nnan ninf nsz arcp afn float %{{.*}}, %{{.*}}
// LLVM-FAST-NO-CONTRACT: fsub reassoc nnan ninf nsz arcp afn float %{{.*}}, %{{.*}}
// LLVM-FAST-NO-CONTRACT: fmul reassoc nnan ninf nsz arcp afn float %{{.*}}, %{{.*}}
// LLVM-FAST-NO-CONTRACT: fdiv reassoc nnan ninf nsz arcp afn float %{{.*}}, %{{.*}}
// CIR-FAST: cir.fadd %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [fast]>{{( loc.*)?$}}
// CIR-FAST: cir.fsub %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [fast]>{{( loc.*)?$}}
// CIR-FAST: cir.fmul %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [fast]>{{( loc.*)?$}}
// CIR-FAST: cir.fdiv %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [fast]>{{( loc.*)?$}}
// LLVM-FAST: fadd fast float %{{.*}}, %{{.*}}
// LLVM-FAST: fsub fast float %{{.*}}, %{{.*}}
// LLVM-FAST: fmul fast float %{{.*}}, %{{.*}}
// LLVM-FAST: fdiv fast float %{{.*}}, %{{.*}}

typedef float float4 __attribute__((ext_vector_type(4)));
float4 vector(float4 a, float4 b, float4 c) {
  return ((a + b) * (a - b)) / c;
}
// CIR-LABEL: cir.func {{.*}}@vector(
// LLVM-LABEL: define {{.*}}@vector(
// CIR-NONE: cir.fadd %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>{{( loc.*)?$}}
// CIR-NONE: cir.fsub %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>{{( loc.*)?$}}
// CIR-NONE: cir.fmul %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>{{( loc.*)?$}}
// CIR-NONE: cir.fdiv %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>{{( loc.*)?$}}
// LLVM-NONE: fadd <4 x float> %{{.*}}, %{{.*}}
// LLVM-NONE: fsub <4 x float> %{{.*}}, %{{.*}}
// LLVM-NONE: fmul <4 x float> %{{.*}}, %{{.*}}
// LLVM-NONE: fdiv <4 x float> %{{.*}}, %{{.*}}
// CIR-NNAN: cir.fadd %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [nnan]>{{( loc.*)?$}}
// CIR-NNAN: cir.fsub %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [nnan]>{{( loc.*)?$}}
// CIR-NNAN: cir.fmul %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [nnan]>{{( loc.*)?$}}
// CIR-NNAN: cir.fdiv %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [nnan]>{{( loc.*)?$}}
// LLVM-NNAN: fadd nnan <4 x float> %{{.*}}, %{{.*}}
// LLVM-NNAN: fsub nnan <4 x float> %{{.*}}, %{{.*}}
// LLVM-NNAN: fmul nnan <4 x float> %{{.*}}, %{{.*}}
// LLVM-NNAN: fdiv nnan <4 x float> %{{.*}}, %{{.*}}
// CIR-NINF: cir.fadd %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [ninf]>{{( loc.*)?$}}
// CIR-NINF: cir.fsub %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [ninf]>{{( loc.*)?$}}
// CIR-NINF: cir.fmul %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [ninf]>{{( loc.*)?$}}
// CIR-NINF: cir.fdiv %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [ninf]>{{( loc.*)?$}}
// LLVM-NINF: fadd ninf <4 x float> %{{.*}}, %{{.*}}
// LLVM-NINF: fsub ninf <4 x float> %{{.*}}, %{{.*}}
// LLVM-NINF: fmul ninf <4 x float> %{{.*}}, %{{.*}}
// LLVM-NINF: fdiv ninf <4 x float> %{{.*}}, %{{.*}}
// CIR-NSZ: cir.fadd %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [nsz]>{{( loc.*)?$}}
// CIR-NSZ: cir.fsub %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [nsz]>{{( loc.*)?$}}
// CIR-NSZ: cir.fmul %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [nsz]>{{( loc.*)?$}}
// CIR-NSZ: cir.fdiv %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [nsz]>{{( loc.*)?$}}
// LLVM-NSZ: fadd nsz <4 x float> %{{.*}}, %{{.*}}
// LLVM-NSZ: fsub nsz <4 x float> %{{.*}}, %{{.*}}
// LLVM-NSZ: fmul nsz <4 x float> %{{.*}}, %{{.*}}
// LLVM-NSZ: fdiv nsz <4 x float> %{{.*}}, %{{.*}}
// CIR-ARCP: cir.fadd %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [arcp]>{{( loc.*)?$}}
// CIR-ARCP: cir.fsub %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [arcp]>{{( loc.*)?$}}
// CIR-ARCP: cir.fmul %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [arcp]>{{( loc.*)?$}}
// CIR-ARCP: cir.fdiv %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [arcp]>{{( loc.*)?$}}
// LLVM-ARCP: fadd arcp <4 x float> %{{.*}}, %{{.*}}
// LLVM-ARCP: fsub arcp <4 x float> %{{.*}}, %{{.*}}
// LLVM-ARCP: fmul arcp <4 x float> %{{.*}}, %{{.*}}
// LLVM-ARCP: fdiv arcp <4 x float> %{{.*}}, %{{.*}}
// CIR-AFN: cir.fadd %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [afn]>{{( loc.*)?$}}
// CIR-AFN: cir.fsub %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [afn]>{{( loc.*)?$}}
// CIR-AFN: cir.fmul %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [afn]>{{( loc.*)?$}}
// CIR-AFN: cir.fdiv %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [afn]>{{( loc.*)?$}}
// LLVM-AFN: fadd afn <4 x float> %{{.*}}, %{{.*}}
// LLVM-AFN: fsub afn <4 x float> %{{.*}}, %{{.*}}
// LLVM-AFN: fmul afn <4 x float> %{{.*}}, %{{.*}}
// LLVM-AFN: fdiv afn <4 x float> %{{.*}}, %{{.*}}
// CIR-REASSOC: cir.fadd %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [reassoc]>{{( loc.*)?$}}
// CIR-REASSOC: cir.fsub %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [reassoc]>{{( loc.*)?$}}
// CIR-REASSOC: cir.fmul %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [reassoc]>{{( loc.*)?$}}
// CIR-REASSOC: cir.fdiv %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [reassoc]>{{( loc.*)?$}}
// LLVM-REASSOC: fadd reassoc <4 x float> %{{.*}}, %{{.*}}
// LLVM-REASSOC: fsub reassoc <4 x float> %{{.*}}, %{{.*}}
// LLVM-REASSOC: fmul reassoc <4 x float> %{{.*}}, %{{.*}}
// LLVM-REASSOC: fdiv reassoc <4 x float> %{{.*}}, %{{.*}}
// CIR-FINITE: cir.fadd %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [nnan, ninf]>{{( loc.*)?$}}
// CIR-FINITE: cir.fsub %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [nnan, ninf]>{{( loc.*)?$}}
// CIR-FINITE: cir.fmul %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [nnan, ninf]>{{( loc.*)?$}}
// CIR-FINITE: cir.fdiv %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [nnan, ninf]>{{( loc.*)?$}}
// LLVM-FINITE: fadd nnan ninf <4 x float> %{{.*}}, %{{.*}}
// LLVM-FINITE: fsub nnan ninf <4 x float> %{{.*}}, %{{.*}}
// LLVM-FINITE: fmul nnan ninf <4 x float> %{{.*}}, %{{.*}}
// LLVM-FINITE: fdiv nnan ninf <4 x float> %{{.*}}, %{{.*}}
// CIR-UNSAFE: cir.fadd %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [nsz, arcp, afn, reassoc]>{{( loc.*)?$}}
// CIR-UNSAFE: cir.fsub %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [nsz, arcp, afn, reassoc]>{{( loc.*)?$}}
// CIR-UNSAFE: cir.fmul %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [nsz, arcp, afn, reassoc]>{{( loc.*)?$}}
// CIR-UNSAFE: cir.fdiv %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [nsz, arcp, afn, reassoc]>{{( loc.*)?$}}
// LLVM-UNSAFE: fadd reassoc nsz arcp afn <4 x float> %{{.*}}, %{{.*}}
// LLVM-UNSAFE: fsub reassoc nsz arcp afn <4 x float> %{{.*}}, %{{.*}}
// LLVM-UNSAFE: fmul reassoc nsz arcp afn <4 x float> %{{.*}}, %{{.*}}
// LLVM-UNSAFE: fdiv reassoc nsz arcp afn <4 x float> %{{.*}}, %{{.*}}
// CIR-FAST-NO-CONTRACT: cir.fadd %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [nnan, ninf, nsz, arcp, afn, reassoc]>{{( loc.*)?$}}
// CIR-FAST-NO-CONTRACT: cir.fsub %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [nnan, ninf, nsz, arcp, afn, reassoc]>{{( loc.*)?$}}
// CIR-FAST-NO-CONTRACT: cir.fmul %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [nnan, ninf, nsz, arcp, afn, reassoc]>{{( loc.*)?$}}
// CIR-FAST-NO-CONTRACT: cir.fdiv %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [nnan, ninf, nsz, arcp, afn, reassoc]>{{( loc.*)?$}}
// LLVM-FAST-NO-CONTRACT: fadd reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, %{{.*}}
// LLVM-FAST-NO-CONTRACT: fsub reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, %{{.*}}
// LLVM-FAST-NO-CONTRACT: fmul reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, %{{.*}}
// LLVM-FAST-NO-CONTRACT: fdiv reassoc nnan ninf nsz arcp afn <4 x float> %{{.*}}, %{{.*}}
// CIR-FAST: cir.fadd %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [fast]>{{( loc.*)?$}}
// CIR-FAST: cir.fsub %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [fast]>{{( loc.*)?$}}
// CIR-FAST: cir.fmul %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [fast]>{{( loc.*)?$}}
// CIR-FAST: cir.fdiv %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> <fastmath_flags = [fast]>{{( loc.*)?$}}
// LLVM-FAST: fadd fast <4 x float> %{{.*}}, %{{.*}}
// LLVM-FAST: fsub fast <4 x float> %{{.*}}, %{{.*}}
// LLVM-FAST: fmul fast <4 x float> %{{.*}}, %{{.*}}
// LLVM-FAST: fdiv fast <4 x float> %{{.*}}, %{{.*}}
