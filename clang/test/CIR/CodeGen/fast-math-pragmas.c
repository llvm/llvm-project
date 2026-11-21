// Verify nested pragma changes and restoration of all fast-math flags.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR,CIR-PRECISE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffp-contract=off -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-PRECISE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffp-contract=off -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-PRECISE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffast-math -ffp-contract=fast -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR,CIR-FAST
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -ffast-math -ffp-contract=fast -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-FAST
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffast-math -ffp-contract=fast -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-FAST

float nested(float a, float b) {
  float r = a / b;
  {
#pragma float_control(precise, on)
    r = r / b;
    {
#pragma clang fp reassociate(on) reciprocal(on)
      r = r / b;
      {
#pragma clang fp reciprocal(off)
        r = r / b;
      }
      r = r / b;
    }
    r = r / b;
  }
  return r / b;
}
// CIR-LABEL: cir.func {{.*}}@nested(
// CIR-PRECISE: cir.fdiv %{{.*}}, %{{.*}} : !cir.float{{( loc.*)?$}}
// CIR-FAST: cir.fdiv %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [fast]>
// CIR: cir.fdiv %{{.*}}, %{{.*}} : !cir.float{{( loc.*)?$}}
// CIR: cir.fdiv %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [arcp, reassoc]>
// CIR: cir.fdiv %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [reassoc]>
// CIR: cir.fdiv %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [arcp, reassoc]>
// CIR: cir.fdiv %{{.*}}, %{{.*}} : !cir.float{{( loc.*)?$}}
// CIR-PRECISE: cir.fdiv %{{.*}}, %{{.*}} : !cir.float{{( loc.*)?$}}
// CIR-FAST: cir.fdiv %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [fast]>
// LLVM-LABEL: define {{.*}}@nested(
// LLVM-PRECISE: fdiv float
// LLVM-FAST: fdiv fast float
// LLVM: fdiv float
// LLVM: fdiv reassoc arcp float
// LLVM: fdiv reassoc float
// LLVM: fdiv reassoc arcp float
// LLVM: fdiv float
// LLVM-PRECISE: fdiv float
// LLVM-FAST: fdiv fast float

// State also resets between functions, including after a function whose
// entire body overrides the command-line options.
float precise(float a, float b) {
#pragma float_control(precise, on)
  return a + b;
}
// CIR-LABEL: cir.func {{.*}}@precise(
// CIR: cir.fadd %{{.*}}, %{{.*}} : !cir.float{{( loc.*)?$}}
// LLVM-LABEL: define {{.*}}@precise(
// LLVM: fadd float

float restored(float a, float b) {
  return a + b;
}
// CIR-LABEL: cir.func {{.*}}@restored(
// CIR-PRECISE: cir.fadd %{{.*}}, %{{.*}} : !cir.float{{( loc.*)?$}}
// CIR-FAST: cir.fadd %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [fast]>
// LLVM-LABEL: define {{.*}}@restored(
// LLVM-PRECISE: fadd float
// LLVM-FAST: fadd fast float

// Enabling imprecise arithmetic sets all fast-math flags even when
// the command line starts with precise arithmetic, and restores them on exit.
float imprecise(float a, float b) {
  float r = a / b;
  {
#pragma float_control(precise, off)
    r = r / b;
  }
  return r / b;
}
// CIR-LABEL: cir.func {{.*}}@imprecise(
// CIR-PRECISE: cir.fdiv %{{.*}}, %{{.*}} : !cir.float{{( loc.*)?$}}
// CIR-FAST: cir.fdiv %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [fast]>
// CIR: cir.fdiv %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [fast]>
// CIR-PRECISE: cir.fdiv %{{.*}}, %{{.*}} : !cir.float{{( loc.*)?$}}
// CIR-FAST: cir.fdiv %{{.*}}, %{{.*}} : !cir.float <fastmath_flags = [fast]>
// LLVM-LABEL: define {{.*}}@imprecise(
// LLVM-PRECISE: fdiv float
// LLVM-FAST: fdiv fast float
// LLVM: fdiv fast float
// LLVM-PRECISE: fdiv float
// LLVM-FAST: fdiv fast float
