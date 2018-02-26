// RUN: %clang -std=c++11 --target=aarch64-arm--eabi -S -emit-llvm %s -o - | FileCheck %s  --check-prefix=CHECK --check-prefix=CHECK-AARCH64
// RUN: %clang -std=c++11 --target=x86_64 -S -emit-llvm %s -o - | FileCheck %s  --check-prefix=CHECK --check-prefix=CHECK-X86

/*  Various contexts where type _Float16 can appear. */


/*  Namespace */

namespace {
  _Float16 f1n;
// CHECK-DAG: @_ZN12_GLOBAL__N_13f1nE = internal global half 0xH0000, align 2

  _Float16 f2n = 33.f16;
// CHECK-DAG: @_ZN12_GLOBAL__N_13f2nE = internal global half 0xH5020, align 2

  _Float16 arr1n[10];
// CHECK-AARCH64-DAG: @_ZN12_GLOBAL__N_15arr1nE = internal global [10 x half] zeroinitializer, align 2
// CHECK-X86-DAG:     @_ZN12_GLOBAL__N_15arr1nE = internal global [10 x half] zeroinitializer, align 16

  _Float16 arr2n[] = { 1.2, 3.0, 3.e4 };
// CHECK-DAG: @_ZN12_GLOBAL__N_15arr2nE = internal global [3 x half] [half 0xH3CCD, half 0xH4200, half 0xH7753], align 2

  const volatile _Float16 func1n(const _Float16 &arg) {
    return arg + f2n + arr1n[4] - arr2n[1];
  }
}


/* File */

_Float16 f1f;
// CHECK-AARCH64-DAG: @f1f = global half 0xH0000, align 2
// CHECK-X86-DAG: @f1f = global half 0xH0000, align 2

_Float16 f2f = 32.4;
// CHECK-DAG: @f2f = global half 0xH500D, align 2

_Float16 arr1f[10];
// CHECK-AARCH64-DAG: @arr1f = global [10 x half] zeroinitializer, align 2
// CHECK-X86-DAG: @arr1f = global [10 x half] zeroinitializer, align 16

_Float16 arr2f[] = { -1.2, -3.0, -3.e4 };
// CHECK-DAG: @arr2f = global [3 x half] [half 0xHBCCD, half 0xHC200, half 0xHF753], align 2

_Float16 func1f(_Float16 arg);


/* Class */

class C1 {
  _Float16 f1c;

  static const _Float16 f2c;
// CHECK-DAG: @_ZN2C13f2cE = external constant half, align 2

  volatile _Float16 f3c;

public:
  C1(_Float16 arg) : f1c(arg), f3c(arg) { }
// Check that we mangle _Float16 to DF16_
// CHECK-DAG: define linkonce_odr void @_ZN2C1C2EDF16_(%class.C1*{{.*}}, half{{.*}})

  _Float16 func1c(_Float16 arg ) {
    return f1c + arg;
  }
// CHECK-DAG: define linkonce_odr half @_ZN2C16func1cEDF16_(%class.C1*{{.*}}, half{{.*}})

  static _Float16 func2c(_Float16 arg) {
    return arg * C1::f2c;
  }
// CHECK-DAG: define linkonce_odr half @_ZN2C16func2cEDF16_(half{{.*}})
};

/*  Template */

template <class C> C func1t(C arg) {
  return arg * 2.f16;
}
// CHECK-DAG: define linkonce_odr half @_Z6func1tIDF16_ET_S0_(half{{.*}})

template <class C> struct S1 {
  C mem1;
};

template <> struct S1<_Float16> {
  _Float16 mem2;
};


/* Local */

extern int printf (const char *__restrict __format, ...);

int main(void) {
  _Float16 f1l = 1e3f16;
// CHECK-DAG: store half 0xH63D0, half* %{{.*}}, align 2

  _Float16 f2l = -0.f16;
// CHECK-DAG: store half 0xH8000, half* %{{.*}}, align 2

  _Float16 f3l = 1.000976562;
// CHECK-DAG: store half 0xH3C01, half* %{{.*}}, align 2

  C1 c1(f1l);
// CHECK-DAG:  [[F1L:%[a-z0-9]+]] = load half, half* %{{.*}}, align 2
// CHECK-DAG:  call void @_ZN2C1C2EDF16_(%class.C1* %{{.*}}, half %{{.*}})

  S1<_Float16> s1 = { 132.f16 };
// CHECK-DAG: @_ZZ4mainE2s1 = private unnamed_addr constant %struct.S1 { half 0xH5820 }, align 2
// CHECK-DAG:  [[S1:%[0-9]+]] = bitcast %struct.S1* %{{.*}} to i8*
// CHECK-DAG: call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[S1]], i8* bitcast (%struct.S1* @_ZZ4mainE2s1 to i8*), i64 2, i32 2, i1 false)

  _Float16 f4l = func1n(f1l)  + func1f(f2l) + c1.func1c(f3l) + c1.func2c(f1l) +
    func1t(f1l) + s1.mem2 - f1n + f2n;

  auto f5l = -1.f16, *f6l = &f2l, f7l = func1t(f3l);
// CHECK-DAG:  store half 0xHBC00, half* %{{.*}}, align 2
// CHECK-DAG:  store half* %{{.*}}, half** %{{.*}}, align 8

  _Float16 f8l = f4l++;
// CHECK-DAG:  %{{.*}} = load half, half* %{{.*}}, align 2
// CHECK-DAG:  [[INC:%[a-z0-9]+]] = fadd half {{.*}}, 0xH3C00
// CHECK-DAG:  store half [[INC]], half* %{{.*}}, align 2

  _Float16 arr1l[] = { -1.f16, -0.f16, -11.f16 };
// CHECK-DAG: @_ZZ4mainE5arr1l = private unnamed_addr constant [3 x half] [half 0xHBC00, half 0xH8000, half 0xHC980], align 2

  float cvtf = f2n;
//CHECK-DAG: [[H2F:%[a-z0-9]+]] = fpext half {{%[0-9]+}} to float
//CHECK-DAG:  store float [[H2F]], float* %{{.*}}, align 4

  double cvtd = f2n;
//CHECK-DAG: [[H2D:%[a-z0-9]+]] = fpext half {{%[0-9]+}} to double
//CHECK-DAG: store double [[H2D]], double* %{{.*}}, align 8


  long double cvtld = f2n;
//CHECK-AARCh64-DAG: [[H2LD:%[a-z0-9]+]] = fpext half {{%[0-9]+}} to fp128
//CHECK-AARCh64-DAG: store fp128 [[H2LD]], fp128* %{{.*}}, align 16
//CHECK-X86-DAG:     [[H2LD:%[a-z0-9]+]] = fpext half {{%[0-9]+}} to x86_fp80
//CHECK-X86-DAG:     store x86_fp80 [[H2LD]], x86_fp80* %{{.*}}, align 16

  _Float16 f2h = 42.0f;
//CHECK-DAG: store half 0xH5140, half* %{{.*}}, align 2
  _Float16 d2h = 42.0;
//CHECK-DAG: store half 0xH5140, half* %{{.*}}, align 2
  _Float16 ld2h = 42.0l;
//CHECK-DAG:store half 0xH5140, half* %{{.*}}, align 2
}
