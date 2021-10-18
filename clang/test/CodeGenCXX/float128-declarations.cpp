// RUN: %clang_cc1 -emit-llvm -triple powerpc64-unknown-unknown \
// RUN:   -target-feature +float128 -std=c++11 %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -triple powerpc64le-unknown-unknown \
// RUN:   -target-feature +float128 -std=c++11 %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -triple i386-unknown-linux-gnu -std=c++11 \
// RUN:   %s -o - | FileCheck %s -check-prefix=CHECK-X86
// RUN: %clang_cc1 -emit-llvm -triple x86_64-unknown-linux-gnu -std=c++11 \
// RUN:   %s -o - | FileCheck %s -check-prefix=CHECK-X86
// RUN: %clang_cc1 -emit-llvm -triple i686-pc-openbsd -std=c++11 \
// RUN:   %s -o - | FileCheck %s -check-prefix=CHECK-X86
// RUN: %clang_cc1 -emit-llvm -triple amd64-pc-openbsd -std=c++11 \
// RUN:   %s -o - | FileCheck %s -check-prefix=CHECK-X86
// RUN: %clang_cc1 -emit-llvm -triple i386-pc-solaris2.11 -std=c++11 \
// RUN:   %s -o - | FileCheck %s -check-prefix=CHECK-X86
// RUN: %clang_cc1 -emit-llvm -triple x86_64-pc-solaris2.11 -std=c++11 \
// RUN:   %s -o - | FileCheck %s -check-prefix=CHECK-X86
// RUN: %clang_cc1 -emit-llvm -triple i586-pc-haiku -std=c++11 \
// RUN:   %s -o - | FileCheck %s -check-prefix=CHECK-X86
// RUN: %clang_cc1 -emit-llvm -triple x86_64-unknown-haiku -std=c++11 \
// RUN:   %s -o - | FileCheck %s -check-prefix=CHECK-X86
//
/*  Various contexts where type __float128 can appear. The different check
    prefixes are due to different mangling on X86.  */

/*  Namespace */
namespace {
  __float128 f1n;
  __float128 f2n = 33.q;
  __float128 arr1n[10];
  __float128 arr2n[] = { 1.2q, 3.0q, 3.e11q };
  const volatile __float128 func1n(const __float128 &arg) {
    return arg + f2n + arr1n[4] - arr2n[1];
  }
}

/* File */
__float128 f1f;
__float128 f2f = 32.4q;
static __float128 f3f = f2f;
__float128 arr1f[10];
__float128 arr2f[] = { -1.2q, -3.0q, -3.e11q };
__float128 func1f(__float128 arg);

/* Class */
class C1 {
  __float128 f1c;
  static const __float128 f2c;
  volatile __float128 f3c;
public:
  C1(__float128 arg) : f1c(arg), f3c(arg) { }
  __float128 func1c(__float128 arg ) {
    return f1c + arg;
  }
  static __float128 func2c(__float128 arg) {
    return arg * C1::f2c;
  }
};

/*  Template */
template <class C> C func1t(C arg) { return arg * 2.q; }
template <class C> struct S1 {
  C mem1;
};
template <> struct S1<__float128> {
  __float128 mem2;
};

/* Local */
int main(void) {
  __float128 f1l = 123e220q;
  __float128 f2l = -0.q;
  __float128 f3l = 1.189731495357231765085759326628007e4932q;
  C1 c1(f1l);
  S1<__float128> s1 = { 132.q };
  __float128 f4l = func1n(f1l) + func1f(f2l) + c1.func1c(f3l) + c1.func2c(f1l) +
    func1t(f1l) + s1.mem2 - f1n + f2n;
#if (__cplusplus >= 201103L)
  auto f5l = -1.q, *f6l = &f2l, f7l = func1t(f3l);
#endif
  __float128 f8l = f4l++;
  __float128 arr1l[] = { -1.q, -0.q, -11.q };
}
// CHECK-DAG: @_ZN12_GLOBAL__N_13f1nE = internal global fp128 0xL00000000000000000000000000000000
// CHECK-DAG: @_ZN12_GLOBAL__N_13f2nE = internal global fp128 0xL00000000000000004004080000000000
// CHECK-DAG: @_ZN12_GLOBAL__N_15arr1nE = internal global [10 x fp128]
// CHECK-DAG: @_ZN12_GLOBAL__N_15arr2nE = internal global [3 x fp128] [fp128 0xL33333333333333333FFF333333333333, fp128 0xL00000000000000004000800000000000, fp128 0xL00000000000000004025176592E00000]
// CHECK-DAG: define internal fp128 @_ZN12_GLOBAL__N_16func1nERKu9__ieee128(fp128*
// CHECK-DAG: @f1f ={{.*}} global fp128 0xL00000000000000000000000000000000
// CHECK-DAG: @f2f ={{.*}} global fp128 0xL33333333333333334004033333333333
// CHECK-DAG: @arr1f ={{.*}} global [10 x fp128]
// CHECK-DAG: @arr2f ={{.*}} global [3 x fp128] [fp128 0xL3333333333333333BFFF333333333333, fp128 0xL0000000000000000C000800000000000, fp128 0xL0000000000000000C025176592E00000]
// CHECK-DAG: declare fp128 @_Z6func1fu9__ieee128(fp128)
// CHECK-DAG: define linkonce_odr void @_ZN2C1C2Eu9__ieee128(%class.C1* {{[^,]*}} %this, fp128 %arg)
// CHECK-DAG: define linkonce_odr fp128 @_ZN2C16func2cEu9__ieee128(fp128 %arg)
// CHECK-DAG: define linkonce_odr fp128 @_Z6func1tIu9__ieee128ET_S0_(fp128 %arg)
// CHECK-DAG: @__const.main.s1 = private unnamed_addr constant %struct.S1 { fp128 0xL00000000000000004006080000000000 }
// CHECK-DAG: store fp128 0xLF0AFD0EBFF292DCE42E0B38CDD83F26F, fp128* %f1l, align 16
// CHECK-DAG: store fp128 0xL00000000000000008000000000000000, fp128* %f2l, align 16
// CHECK-DAG: store fp128 0xLFFFFFFFFFFFFFFFF7FFEFFFFFFFFFFFF, fp128* %f3l, align 16
// CHECK-DAG: store fp128 0xL0000000000000000BFFF000000000000, fp128* %f5l, align 16
// CHECK-DAG: [[F4L:%[a-z0-9]+]] = load fp128, fp128* %f4l
// CHECK-DAG: [[INC:%[a-z0-9]+]] = fadd fp128 [[F4L]], 0xL00000000000000003FFF000000000000
// CHECK-DAG: store fp128 [[INC]], fp128* %f4l

// CHECK-X86-DAG: @_ZN12_GLOBAL__N_13f1nE = internal global fp128 0xL00000000000000000000000000000000
// CHECK-X86-DAG: @_ZN12_GLOBAL__N_13f2nE = internal global fp128 0xL00000000000000004004080000000000
// CHECK-X86-DAG: @_ZN12_GLOBAL__N_15arr1nE = internal global [10 x fp128]
// CHECK-X86-DAG: @_ZN12_GLOBAL__N_15arr2nE = internal global [3 x fp128] [fp128 0xL33333333333333333FFF333333333333, fp128 0xL00000000000000004000800000000000, fp128 0xL00000000000000004025176592E00000]
// CHECK-X86-DAG: define internal fp128 @_ZN12_GLOBAL__N_16func1nERKg(fp128*
// CHECK-X86-DAG: @f1f ={{.*}} global fp128 0xL00000000000000000000000000000000
// CHECK-X86-DAG: @f2f ={{.*}} global fp128 0xL33333333333333334004033333333333
// CHECK-X86-DAG: @arr1f ={{.*}} global [10 x fp128]
// CHECK-X86-DAG: @arr2f ={{.*}} global [3 x fp128] [fp128 0xL3333333333333333BFFF333333333333, fp128 0xL0000000000000000C000800000000000, fp128 0xL0000000000000000C025176592E00000]
// CHECK-X86-DAG: declare fp128 @_Z6func1fg(fp128)
// CHECK-X86-DAG: define linkonce_odr void @_ZN2C1C2Eg(%class.C1* {{[^,]*}} %this, fp128 %arg)
// CHECK-X86-DAG: define linkonce_odr fp128 @_ZN2C16func2cEg(fp128 %arg)
// CHECK-X86-DAG: define linkonce_odr fp128 @_Z6func1tIgET_S0_(fp128 %arg)
// CHECK-X86-DAG: @__const.main.s1 = private unnamed_addr constant %struct.S1 { fp128 0xL00000000000000004006080000000000 }
// CHECK-X86-DAG: store fp128 0xLF0AFD0EBFF292DCE42E0B38CDD83F26F, fp128* %f1l, align 16
// CHECK-X86-DAG: store fp128 0xL00000000000000008000000000000000, fp128* %f2l, align 16
// CHECK-X86-DAG: store fp128 0xLFFFFFFFFFFFFFFFF7FFEFFFFFFFFFFFF, fp128* %f3l, align 16
// CHECK-X86-DAG: store fp128 0xL0000000000000000BFFF000000000000, fp128* %f5l, align 16
// CHECK-X86-DAG: [[F4L:%[a-z0-9]+]] = load fp128, fp128* %f4l
// CHECK-X86-DAG: [[INC:%[a-z0-9]+]] = fadd fp128 [[F4L]], 0xL00000000000000003FFF000000000000
// CHECK-X86-DAG: store fp128 [[INC]], fp128* %f4l
