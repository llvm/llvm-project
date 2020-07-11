// RUN: %clang_cc1 -ftrapping-math -frounding-math -ffp-exception-behavior=strict -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=FPMODELSTRICT
// RUN: %clang_cc1 -ffp-contract=fast -emit-llvm -o - %s | FileCheck %s -check-prefix=PRECISE
// RUN: %clang_cc1 -ffast-math -ffp-contract=fast -emit-llvm -o - %s | FileCheck %s -check-prefix=FAST
// RUN: %clang_cc1 -ffast-math -emit-llvm -o - %s | FileCheck %s -check-prefix=FASTNOCONTRACT
// RUN: %clang_cc1 -ffast-math -ffp-contract=fast -ffp-exception-behavior=ignore -emit-llvm -o - %s | FileCheck %s -check-prefix=FAST
// RUN: %clang_cc1 -ffast-math -ffp-contract=fast -ffp-exception-behavior=strict -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=EXCEPT
// RUN: %clang_cc1 -ffast-math -ffp-contract=fast -ffp-exception-behavior=maytrap -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=MAYTRAP

float f0, f1, f2;

void foo() {
  // CHECK-LABEL: define {{.*}}void @foo()

  // MAYTRAP: llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
  // EXCEPT: llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
  // FPMODELSTRICT: llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  // STRICTEXCEPT: llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  // STRICTNOEXCEPT: llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
  // PRECISE: fadd contract float %{{.*}}, %{{.*}}
  // FAST: fadd fast
  // FASTNOCONTRACT: fadd reassoc nnan ninf nsz arcp afn float
  f0 = f1 + f2;

  // CHECK: ret
}
