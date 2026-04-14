// RUN: %clang_cc1 -triple riscv64 -frounding-math -ffp-exception-behavior=strict -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=FPMODELSTRICT
// RUN: %clang_cc1 -triple riscv64 -ffp-contract=fast -emit-llvm -o - %s | FileCheck %s -check-prefix=PRECISE
// RUN: %clang_cc1 -triple riscv64 -ffast-math -ffp-contract=fast -emit-llvm -o - %s | FileCheck %s -check-prefix=FAST
// RUN: %clang_cc1 -triple riscv64 -ffast-math -emit-llvm -o - %s | FileCheck %s -check-prefix=FASTNOCONTRACT
// RUN: %clang_cc1 -triple riscv64 -ffast-math -ffp-contract=fast -ffp-exception-behavior=ignore -emit-llvm -o - %s | FileCheck %s -check-prefix=FAST
// RUN: %clang_cc1 -triple riscv64 -ffast-math -ffp-contract=fast -ffp-exception-behavior=strict -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=EXCEPT
// RUN: %clang_cc1 -triple riscv64 -ffast-math -ffp-contract=fast -ffp-exception-behavior=maytrap -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=MAYTRAP

// Test strict-fp support in RISC-V.

float f0, f1, f2;

void foo(void) {
  // CHECK-LABEL: define {{.*}}void @foo()

  // MAYTRAP: call fast float @llvm.fadd.f32(float %{{.*}}, float %{{.*}}) {{.*}}[ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // EXCEPT: call fast float @llvm.fadd.f32(float %{{.*}}, float %{{.*}}) {{.*}}[ "fp.control"(metadata !"rte") ]
  // FPMODELSTRICT: fadd float %{{.*}}, %{{.*}}
  // STRICTEXCEPT: llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
  // STRICTNOEXCEPT: llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
  // PRECISE: fadd contract float %{{.*}}, %{{.*}}
  // FAST: fadd fast
  // FASTNOCONTRACT: fadd reassoc nnan ninf nsz arcp afn float
  f0 = f1 + f2;

  // CHECK: ret
}
