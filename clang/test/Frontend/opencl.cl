// RUN: %clang_cc1 %s -verify -fsyntax-only -DSYNTAX
// RUN: %clang_cc1 %s -verify -fsyntax-only -cl-std=CL1.1 -DSYNTAX
// RUN: %clang_cc1 %s -verify -fsyntax-only -cl-std=CL1.2 -DSYNTAX
// RUN: %clang_cc1 %s -verify -fsyntax-only -cl-std=CL2.0 -DSYNTAX
// RUN: %clang_cc1 %s -verify -fsyntax-only -cl-std=clc++1.0 -DSYNTAX
// RUN: %clang_cc1 %s -verify -fsyntax-only -fblocks -DBLOCKS -DSYNTAX
// RUN: %clang_cc1 %s -verify -fsyntax-only -cl-std=CL1.1 -fblocks -DBLOCKS -DSYNTAX
// RUN: %clang_cc1 %s -verify -fsyntax-only -cl-std=CL1.2 -fblocks -DBLOCKS -DSYNTAX
// RUN: %clang_cc1 %s -triple amdgcn--amdhsa -x c -std=c99 -verify -fsyntax-only -DSYNTAX
// RUN: %clang_cc1 -cl-std=CL1.1 -cl-strict-aliasing -fblocks %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID-OPENCL-VERSION11 %s
// RUN: %clang_cc1 -cl-std=CL1.2 -cl-strict-aliasing -fblocks %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID-OPENCL-VERSION12 %s
// RUN: %clang_cc1 -cl-std=CL2.0 -cl-strict-aliasing %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID-OPENCL-VERSION20 %s
// RUN: %clang_cc1 -cl-std=clc++1.0 -cl-strict-aliasing -fblocks %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID-OPENCLCPP-VERSION10 %s
// RUN: %clang_cc1 %s -verify -fsyntax-only -cl-std=CL3.0 -cl-ext=-__opencl_c_device_enqueue -DSYNTAX

#ifdef SYNTAX
class test{
int member;
};
#ifndef __OPENCL_CPP_VERSION__
//expected-error@-4{{unknown type name 'class'}}
//expected-error@-5{{expected ';' after top level declarator}}
#endif
#endif

typedef int (^bl_t)(void);
#if defined(__OPENCL_C_VERSION__) || defined(__OPENCL_CPP_VERSION__)
#if !defined(BLOCKS) && (defined(__OPENCL_CPP_VERSION__)  || __OPENCL_C_VERSION__ != CL_VERSION_2_0)
  // expected-error@-3{{blocks support disabled - compile with -fblocks or for OpenCL C 2.0 or OpenCL C 3.0 with __opencl_c_device_enqueue feature}}
#endif
#else
  // expected-error@-6{{blocks support disabled - compile with -fblocks or pick a deployment target that supports them}}
#endif

// CHECK-INVALID-OPENCL-VERSION11: warning: OpenCL C version 1.1 does not support the option '-cl-strict-aliasing'
// CHECK-INVALID-OPENCL-VERSION12: warning: OpenCL C version 1.2 does not support the option '-cl-strict-aliasing'
// CHECK-INVALID-OPENCL-VERSION20: warning: OpenCL C version 2.0 does not support the option '-cl-strict-aliasing'
// CHECK-INVALID-OPENCLCPP-VERSION10: warning: C++ for OpenCL version 1.0 does not support the option '-cl-strict-aliasing'
