// RUN: %clang_cc1 -x c++ -fexceptions -fcxx-exceptions -frounding-math -ffp-exception-behavior=strict -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=FPMODELSTRICT
// RUN: %clang_cc1 -x c++ -ffp-contract=fast -fexceptions -fcxx-exceptions -emit-llvm -o - %s | FileCheck %s -check-prefix=PRECISE
// RUN: %clang_cc1 -x c++ -ffast-math -fexceptions -fcxx-exceptions -ffp-contract=fast -emit-llvm -o - %s | FileCheck %s -check-prefix=FAST
// RUN: %clang_cc1 -x c++ -ffast-math -fexceptions -fcxx-exceptions -emit-llvm -o - %s | FileCheck %s -check-prefix=FASTNOCONTRACT
// RUN: %clang_cc1 -x c++ -ffast-math -fexceptions -fcxx-exceptions -ffp-contract=fast -ffp-exception-behavior=ignore -emit-llvm -o - %s | FileCheck %s -check-prefix=FAST
// RUN: %clang_cc1 -x c++ -ffast-math -fexceptions -fcxx-exceptions -ffp-contract=fast -ffp-exception-behavior=strict -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=EXCEPT
// RUN: %clang_cc1 -x c++ -ffast-math -fexceptions -fcxx-exceptions -ffp-contract=fast -ffp-exception-behavior=maytrap -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=MAYTRAP

float f0, f1, f2;

  template <class>
  class aaaa {
   public:
    ~aaaa();
    void b();
  };
  
  template <class c>
  aaaa<c>::~aaaa() { try {
    b();
  // CHECK-LABEL: define {{.*}}void @_ZN4aaaaIiED2Ev{{.*}}

  } catch (...) {
    // MAYTRAP: llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
    // EXCEPT: llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
    // FPMODELSTRICT: llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
    // STRICTEXCEPT: llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
    // STRICTNOEXCEPT: llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
    // PRECISE: fadd contract float %{{.*}}, %{{.*}}
    // FAST: fadd fast
    // FASTNOCONTRACT: fadd reassoc nnan ninf nsz arcp afn float
    f0 = f1 + f2;

    // CHECK: ret void
  }
  }
  
  class d {
   public:
    d(const char *, int);
    aaaa<int> e;
  };
  
float foo() {
  d x("", 1);
  aaaa<int> a;
  return f0;
}

