// RUN: %clang_cc1 %s -verify -fsyntax-only
// RUN: %clang_cc1 %s -verify -fsyntax-only -cl-std=CL1.1
// RUN: %clang_cc1 %s -verify -fsyntax-only -cl-std=CL1.2
// RUN: %clang_cc1 %s -verify -fsyntax-only -cl-std=CL2.0
// RUN: %clang_cc1 %s -verify -fsyntax-only -fblocks -DBLOCKS
// RUN: %clang_cc1 %s -verify -fsyntax-only -cl-std=CL1.1 -fblocks -DBLOCKS
// RUN: %clang_cc1 %s -verify -fsyntax-only -cl-std=CL1.2 -fblocks -DBLOCKS
// RUN: %clang_cc1 %s -triple amdgcn--amdhsa -x c -std=c99 -verify -fsyntax-only
// RUN: %clang_cc1 -cl-std=CL1.1 -cl-strict-aliasing -fblocks %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID-OPENCL-VERSION11 %s
// RUN: %clang_cc1 -cl-std=CL1.2 -cl-strict-aliasing -fblocks %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID-OPENCL-VERSION12 %s
// RUN: %clang_cc1 -cl-std=CL2.0 -cl-strict-aliasing %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID-OPENCL-VERSION20 %s

void f(void (^g)(void)) {
#ifdef __OPENCL_C_VERSION__
#if __OPENCL_C_VERSION__ < CL_VERSION_2_0 && !defined(BLOCKS)
  // expected-error@-3{{blocks support disabled - compile with -fblocks or for OpenCL 2.0 or above}}
#else
  // expected-no-diagnostics
#endif
#else
  // expected-error@-8{{blocks support disabled - compile with -fblocks or pick a deployment target that supports them}}
#endif
}

// CHECK-INVALID-OPENCL-VERSION11: warning: OpenCL version 1.1 does not support the option '-cl-strict-aliasing'
// CHECK-INVALID-OPENCL-VERSION12: warning: OpenCL version 1.2 does not support the option '-cl-strict-aliasing'
// CHECK-INVALID-OPENCL-VERSION20: warning: OpenCL version 2.0 does not support the option '-cl-strict-aliasing'
