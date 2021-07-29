// RUN: %clang_cc1 -triple x86_64-linux-gnu -ffp-contract=on -DDEFAULT=1 -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-DDEFAULT %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -ffp-contract=on -DEBSTRICT=1 -ffp-exception-behavior=strict -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-DEBSTRICT %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -DFAST=1 -ffast-math -ffp-contract=fast -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-FAST %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -ffp-contract=on -DNOHONOR=1 -menable-no-infs -menable-no-nans -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-NOHONOR %s

#define FUN(n) \
  (float z) { return n * z + n; }

// CHECK-DDEFAULT Function Attrs: noinline nounwind optnone mustprogress
// CHECK-DEBSTRICT Function Attrs: noinline nounwind optnone strictfp mustprogress
// CHECK-FAST: Function Attrs: mustprogress noinline nounwind optnone
// CHECK-NOHONOR Function Attrs: noinline nounwind optnone mustprogress
float fun_default FUN(1)
//CHECK-LABEL: define {{.*}} @_Z11fun_defaultf{{.*}}
#if DEFAULT
//CHECK-DDEFAULT: call float @llvm.fmuladd{{.*}}
#endif
#if EBSTRICT
// Note that backend wants constrained intrinsics used
// throughout the function if they are needed anywhere in the function.
// In that case, operations are built with constrained intrinsics operator
// but using default settings for exception behavior and rounding mode.
//CHECK-DEBSTRICT: llvm.experimental.constrained.fmul{{.*}}tonearest{{.*}}strict
#endif
#if FAST
//CHECK-FAST: fmul fast float
//CHECK-FAST: fadd fast float
#endif

#pragma float_control(push)
#ifndef FAST
// Rule: precise must be enabled
#pragma float_control(except, on)
#endif
    // CHECK-FAST: Function Attrs: mustprogress noinline nounwind optnone
    // CHECK-DDEFAULT Function Attrs: noinline nounwind optnone strictfp mustprogress
    // CHECK-DEBSTRICT Function Attrs: noinline nounwind optnone strictfp mustprogress
    // CHECK-NOHONOR Function Attrs: noinline nounwind optnone strictfp mustprogress
    float exc_on FUN(2)
//CHECK-LABEL: define {{.*}} @_Z6exc_onf{{.*}}
#if DEFAULT
//CHECK-DDEFAULT: llvm.experimental.constrained.fmul{{.*}}
#endif
#if EBSTRICT
//CHECK-DEBSTRICT: llvm.experimental.constrained.fmuladd{{.*}}tonearest{{.*}}strict
#endif
#if NOHONOR
//CHECK-NOHONOR: nnan ninf float {{.*}}llvm.experimental.constrained.fmuladd{{.*}}tonearest{{.*}}strict
#endif
#if FAST
//Not possible to enable float_control(except) in FAST mode.
//CHECK-FAST: fmul fast float
//CHECK-FAST: fadd fast float
#endif

#pragma float_control(pop)
    // CHECK-DDEFAULT Function Attrs: noinline nounwind optnone mustprogress
    // CHECK-DEBSTRICT Function Attrs: noinline nounwind optnone strictfp mustprogress
    // CHECK-FAST: Function Attrs: mustprogress noinline nounwind optnone
    // CHECK-NOHONOR Function Attrs: noinline nounwind optnone mustprogress
    float exc_pop FUN(5)
//CHECK-LABEL: define {{.*}} @_Z7exc_popf{{.*}}
#if DEFAULT
//CHECK-DDEFAULT: call float @llvm.fmuladd{{.*}}
#endif
#if EBSTRICT
//CHECK-DEBSTRICT: llvm.experimental.constrained.fmuladd{{.*}}tonearest{{.*}}strict
#endif
#if NOHONOR
//CHECK-NOHONOR: call nnan ninf float @llvm.fmuladd{{.*}}
#endif
#if FAST
//CHECK-FAST: fmul fast float
//CHECK-FAST: fadd fast float
#endif

#pragma float_control(except, off)
        float exc_off FUN(5)
//CHECK-LABEL: define {{.*}} @_Z7exc_offf{{.*}}
#if DEFAULT
//CHECK-DDEFAULT: call float @llvm.fmuladd{{.*}}
#endif
#if EBSTRICT
//CHECK-DEBSTRICT: call float @llvm.fmuladd{{.*}}
#endif
#if NOHONOR
//CHECK-NOHONOR: call nnan ninf float @llvm.fmuladd{{.*}}
#endif
#if FAST
//CHECK-FAST: fmul fast float
//CHECK-FAST: fadd fast float
#endif

#pragma float_control(precise, on, push)
            float precise_on FUN(3)
//CHECK-LABEL: define {{.*}} @_Z10precise_onf{{.*}}
#if DEFAULT
//CHECK-DDEFAULT: float {{.*}}llvm.fmuladd{{.*}}
#endif
#if EBSTRICT
//CHECK-DEBSTRICT: float {{.*}}llvm.fmuladd{{.*}}
#endif
#if NOHONOR
// If precise is pushed then all fast-math should be off!
//CHECK-NOHONOR: call float {{.*}}llvm.fmuladd{{.*}}
#endif
#if FAST
//CHECK-FAST: float {{.*}}llvm.fmuladd{{.*}}
#endif

#pragma float_control(pop)
                float precise_pop FUN(3)
//CHECK-LABEL: define {{.*}} @_Z11precise_popf{{.*}}
#if DEFAULT
//CHECK-DDEFAULT: float {{.*}}llvm.fmuladd{{.*}}
#endif
#if EBSTRICT
//CHECK-DEBSTRICT: float {{.*}}llvm.fmuladd{{.*}}
#endif
#if NOHONOR
//CHECK-NOHONOR: call nnan ninf float @llvm.fmuladd{{.*}}
#endif
#if FAST
//CHECK-FAST: fmul fast float
//CHECK-FAST: fadd fast float
#endif
#pragma float_control(precise, off)
                    float precise_off FUN(4)
//CHECK-LABEL: define {{.*}} @_Z11precise_offf{{.*}}
#if DEFAULT
// Note: precise_off enables fp_contract=fast and the instructions
// generated do not include the contract flag, although it was enabled
// in IRBuilder.
//CHECK-DDEFAULT: fmul fast float
//CHECK-DDEFAULT: fadd fast float
#endif
#if EBSTRICT
//CHECK-DEBSTRICT: fmul fast float
//CHECK-DEBSTRICT: fadd fast float
#endif
#if NOHONOR
// fast math should be enabled, and contract should be fast
//CHECK-NOHONOR: fmul fast float
//CHECK-NOHONOR: fadd fast float
#endif
#if FAST
//CHECK-FAST: fmul fast float
//CHECK-FAST: fadd fast float
#endif

#pragma float_control(precise, on)
                        float precise_on2 FUN(3)
//CHECK-LABEL: define {{.*}} @_Z11precise_on2f{{.*}}
#if DEFAULT
//CHECK-DDEFAULT: llvm.fmuladd{{.*}}
#endif
#if EBSTRICT
//CHECK-DEBSTRICT: float {{.*}}llvm.fmuladd{{.*}}
#endif
#if NOHONOR
// fast math should be off, and contract should be on
//CHECK-NOHONOR: float {{.*}}llvm.fmuladd{{.*}}
#endif
#if FAST
//CHECK-FAST: float {{.*}}llvm.fmuladd{{.*}}
#endif

#pragma float_control(push)
                            float precise_push FUN(3)
//CHECK-LABEL: define {{.*}} @_Z12precise_pushf{{.*}}
#if DEFAULT
//CHECK-DDEFAULT: llvm.fmuladd{{.*}}
#endif
#if EBSTRICT
//CHECK-DEBSTRICT: float {{.*}}llvm.fmuladd{{.*}}
#endif
#if NOHONOR
//CHECK-NOHONOR: float {{.*}}llvm.fmuladd{{.*}}
#endif
#if FAST
//CHECK-FAST: float {{.*}}llvm.fmuladd{{.*}}
#endif

#pragma float_control(precise, off)
                                float precise_off2 FUN(4)
//CHECK-LABEL: define {{.*}} @_Z12precise_off2f{{.*}}
#if DEFAULT
//CHECK-DDEFAULT: fmul fast float
//CHECK-DDEFAULT: fadd fast float
#endif
#if EBSTRICT
//CHECK-DEBSTRICT: fmul fast float
//CHECK-DEBSTRICT: fadd fast float
#endif
#if NOHONOR
// fast math settings since precise is off
//CHECK-NOHONOR: fmul fast float
//CHECK-NOHONOR: fadd fast float
#endif
#if FAST
//CHECK-FAST: fmul fast float
//CHECK-FAST: fadd fast float
#endif

#pragma float_control(pop)
                                    float precise_pop2 FUN(3)
//CHECK-LABEL: define {{.*}} @_Z12precise_pop2f{{.*}}
#if DEFAULT
//CHECK-DDEFAULT: llvm.fmuladd{{.*}}
#endif
#if EBSTRICT
//CHECK-DEBSTRICT: float {{.*}}llvm.fmuladd{{.*}}
#endif
#if NOHONOR
//CHECK-NOHONOR: float {{.*}}llvm.fmuladd{{.*}}
#endif
#if FAST
//CHECK-FAST: float {{.*}}llvm.fmuladd{{.*}}
#endif

#ifndef FAST
// Rule: precise must be enabled
#pragma float_control(except, on)
#endif
                                        float y();
// CHECK-DDEFAULT Function Attrs: noinline nounwind optnone mustprogress
// CHECK-DEBSTRICT Function Attrs: noinline nounwind optnone strictfp mustprogress
// CHECK-FAST: Function Attrs: mustprogress noinline nounwind optnone
// CHECK-NOHONOR Function Attrs: noinline nounwind optnone mustprogress
class ON {
  // Settings for top level class initializer use program source setting.
  float z = 2 + y() * 7;
//CHECK-LABEL: define {{.*}} void @_ZN2ONC2Ev{{.*}}
#if DEFAULT
// CHECK-DDEFAULT: llvm.experimental.constrained.fmul{{.*}}tonearest{{.*}}strict
#endif
#if EBSTRICT
// CHECK-DEBSTRICT: llvm.experimental.constrained.fmul{{.*}}tonearest{{.*}}strict
#endif
#if NOHONOR
// CHECK-NOHONOR: llvm.experimental.constrained.fmul{{.*}}tonearest{{.*}}strict
#endif
#if FAST
// CHECK-FAST: float {{.*}}llvm.fmuladd{{.*}}
#endif
};
ON on;
#pragma float_control(except, off)
// CHECK-DDEFAULT Function Attrs: noinline nounwind optnone
// CHECK-DEBSTRICT Function Attrs: noinline nounwind optnone
// CHECK-FAST: Function Attrs: noinline nounwind optnone
// CHECK-NOHONOR Function Attrs: noinline nounwind optnone
class OFF {
  float w = 2 + y() * 7;
//CHECK-LABEL: define {{.*}} void @_ZN3OFFC2Ev{{.*}}
//CHECK: call float {{.*}}llvm.fmuladd
};
OFF off;

#pragma clang fp reassociate(on)
struct MyComplex {
  float xx;
  float yy;
  MyComplex(float x, float y) {
    xx = x;
    yy = y;
  }
  MyComplex() {}
  const MyComplex operator+(const MyComplex other) const {
//CHECK-LABEL: define {{.*}} @_ZNK9MyComplexplES_
//CHECK: fadd reassoc float
//CHECK: fadd reassoc float
    return MyComplex(xx + other.xx, yy + other.yy);
  }
};
MyComplex useAdd() {
  MyComplex a (1, 3);
  MyComplex b (2, 4);
   return a + b;
}

// CHECK-DDEFAULT Function Attrs: noinline nounwind
// CHECK-DEBSTRICT Function Attrs: noinline nounwind strictfp
// CHECK-FAST: Function Attrs: noinline nounwind
// CHECK-NOHONOR Function Attrs: noinline nounwind
// CHECK-LABEL: define{{.*}} @_GLOBAL__sub_I_fp_floatcontrol_stack
