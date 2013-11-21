// RUN: %clang_cc1 -emit-llvm -cxx-abi microsoft -triple=i386-pc-win32 %s -o - | FileCheck %s
// RUN: %clang_cc1 -DINLINE_INIT -emit-llvm -cxx-abi microsoft -triple=i386-pc-win32 %s -o - | FileCheck %s --check-prefix=CHECK-INLINE
// RUN: %clang_cc1 -DREAL_DEFINITION -emit-llvm -cxx-abi microsoft -triple=i386-pc-win32 %s -o - | FileCheck %s --check-prefix=CHECK-OUTOFLINE
// RUN: %clang_cc1 -DINLINE_INIT -DREAL_DEFINITION -emit-llvm -cxx-abi microsoft -triple=i386-pc-win32 %s -o - | FileCheck %s --check-prefix=CHECK-INLINE

struct S {
  // For MS ABI, we emit a linkonce_odr definition here, even though it's really just a declaration.
#ifdef INLINE_INIT
  static const int x = 5;
#else
  static const int x;
#endif
};

const int *f() {
  return &S::x;
};

#ifdef REAL_DEFINITION
#ifdef INLINE_INIT
const int S::x;
#else
const int S::x = 5;
#endif
#endif


// Inline initialization.
// CHECK-INLINE: @"\01?x@S@@2HB" = linkonce_odr constant i32 5, align 4

// Out-of-line initialization.
// CHECK-OUTOFLINE: @"\01?x@S@@2HB" = constant i32 5, align 4

// No initialization.
// CHECK: @"\01?x@S@@2HB" = external constant i32
