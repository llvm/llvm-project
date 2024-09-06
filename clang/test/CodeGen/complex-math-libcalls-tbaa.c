// RUN: %clang_cc1 %s -O3  -fmath-errno -emit-llvm -triple x86_64-unknown-unknown -o - %s | FileCheck %s -check-prefixes=CHECK
// RUN: %clang_cc1 %s -O3 -fmath-errno -emit-llvm -triple x86_64-pc-win64 -o - %s | FileCheck %s -check-prefixes=CHECK
// RUN: %clang_cc1 %s -O3 -fmath-errno -emit-llvm -triple i686-unknown-unknown -o - %s | FileCheck %s -check-prefixes=CHECK
// RUN: %clang_cc1 %s -O3 -fmath-errno -emit-llvm -triple powerpc-unknown-unknown -o - %s | FileCheck %s -check-prefixes=CHECK-PPC
// RUN: %clang_cc1 %s -O3 -fmath-errno -emit-llvm -triple armv7-none-linux-gnueabi -o - %s | FileCheck %s -check-prefixes=CHECK-TBAA,TBAA
// RUN: %clang_cc1 %s -O3 -fmath-errno -emit-llvm -triple armv7-none-linux-gnueabihf -o - %s | FileCheck %s -check-prefixes=CHECK-ARM,TBAA
// RUN: %clang_cc1 %s -O3 -fmath-errno -emit-llvm -triple thumbv7k-apple-watchos2.0 -o - -target-abi aapcs16 %s | FileCheck %s -check-prefixes=CHECK-THUMB,TBAA
// RUN: %clang_cc1 %s -O3 -fmath-errno -emit-llvm -triple aarch64-unknown-unknown -o - %s | FileCheck %s -check-prefixes=CHECK-AARCH,TBAA
// RUN: %clang_cc1 %s -O3 -fmath-errno -emit-llvm -triple spir -o - %s | FileCheck %s -check-prefixes=CHECK-SPIR

_Complex long double foo() {
  _Complex long double cld;
  long double v2 = __builtin_cargl(cld);
  _Complex long double tmp = v2 * cld;
  return tmp;
}
// CHECK: tail call x86_fp80 @cargl(ptr noundef nonnull byval({ {{.*}}, {{.*}} })
// CHECK-PPC: tail call ppc_fp128 @cargl(ptr noundef nonnull byval({ ppc_fp128, ppc_fp128 })
// CHECK-TBAA: tail call double @cargl([2 x i64] noundef undef) #[[ATTR1:[0-9]+]], !tbaa [[TBAA3:![0-9]+]]

// CHECK-ARM: tail call double @cargl({ double, double } noundef undef) #[[ATTR1:[0-9]+]], !tbaa [[TBAA3:![0-9]+]]

// CHECK-THUMB: tail call double @cargl([2 x double] noundef undef) #[[ATTR1:[0-9]+]], !tbaa [[TBAA3:![0-9]+]]

// CHECK-AARCH: tail call fp128 @cargl([2 x fp128] noundef alignstack(16) undef) #[[ATTR1:[0-9]+]], !tbaa [[TBAA3:![0-9]+]]

// CHECK-SPIR: tail call spir_func double @cargl(ptr noundef nonnull byval({ {{.*}}, {{.*}} })

// TBAA: [[TBAA3]] = !{[[META4:![0-9]+]], [[META4]], i64 0}
// TBAA: [[META4]] = !{!"int", [[META5:![0-9]+]], i64 0}
// TBAA: [[META5]] = !{!"omnipotent char", [[META6:![0-9]+]], i64 0}
