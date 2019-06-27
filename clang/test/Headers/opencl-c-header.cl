// RUN: %clang_cc1 -O0 -triple spir-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -verify | FileCheck %s
// RUN: %clang_cc1 -O0 -triple spir-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -verify -cl-std=CL1.1 | FileCheck %s
// RUN: %clang_cc1 -O0 -triple spir-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -verify -cl-std=CL1.2 | FileCheck %s
// RUN: %clang_cc1 -O0 -triple spir-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -verify -cl-std=c++ | FileCheck %s --check-prefix=CHECK20

// Test including the default header as a module.
// The module should be compiled only once and loaded from cache afterwards.
// Change the directory mode to read only to make sure no new modules are created.
// Check time report to make sure module is used.
// Check that some builtins occur in the generated IR when called.

// ===
// Clear current directory.
// RUN: rm -rf %t
// RUN: mkdir -p %t

// ===
// Compile for OpenCL 1.0 for the first time. A module should be generated.
// RUN: %clang_cc1 -triple spir-unknown-unknown -emit-llvm -o - -finclude-default-header -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash -ftime-report %s 2>&1 | FileCheck --check-prefix=CHECK --check-prefix=CHECK-MOD %s
// RUN: chmod u-w %t/opencl_c.pcm

// ===
// Compile for OpenCL 1.0 for the second time. The module should not be re-created.
// RUN: %clang_cc1 -triple spir-unknown-unknown -emit-llvm -o - -finclude-default-header -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash -ftime-report %s 2>&1 | FileCheck --check-prefix=CHECK --check-prefix=CHECK-MOD %s
// RUN: chmod u+w %t/opencl_c.pcm
// RUN: mv %t/opencl_c.pcm %t/1_0.pcm

// ===
// Compile for OpenCL 2.0 for the first time. The module should change.
// RUN: %clang_cc1 -triple spir-unknown-unknown -O0 -emit-llvm -o - -cl-std=CL2.0 -finclude-default-header -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash -ftime-report %s 2>&1 | FileCheck --check-prefix=CHECK20 --check-prefix=CHECK-MOD %s
// RUN: not diff %t/1_0.pcm %t/opencl_c.pcm
// RUN: chmod u-w %t/opencl_c.pcm

// ===
// Compile for OpenCL 2.0 for the second time. The module should not change.
// RUN: %clang_cc1 -triple spir-unknown-unknown -O0 -emit-llvm -o - -cl-std=CL2.0 -finclude-default-header -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash -ftime-report %s 2>&1 | FileCheck --check-prefix=CHECK20 --check-prefix=CHECK-MOD %s

// Check cached module works for different OpenCL versions.
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: %clang_cc1 -triple spir64-unknown-unknown -emit-llvm -o - -cl-std=CL1.2 -finclude-default-header -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -ftime-report %s 2>&1 | FileCheck --check-prefix=CHECK --check-prefix=CHECK-MOD %s
// RUN: %clang_cc1 -triple amdgcn--amdhsa -O0 -emit-llvm -o - -cl-std=CL2.0 -finclude-default-header -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -ftime-report %s 2>&1 | FileCheck --check-prefix=CHECK20 --check-prefix=CHECK-MOD %s
// RUN: chmod u-w %t 
// RUN: %clang_cc1 -triple spir64-unknown-unknown -emit-llvm -o - -cl-std=CL1.2 -finclude-default-header -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -ftime-report %s 2>&1 | FileCheck --check-prefix=CHECK --check-prefix=CHECK-MOD %s
// RUN: %clang_cc1 -triple amdgcn--amdhsa -O0 -emit-llvm -o - -cl-std=CL2.0 -finclude-default-header -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -ftime-report %s 2>&1 | FileCheck --check-prefix=CHECK20 --check-prefix=CHECK-MOD %s
// RUN: chmod u+w %t

// Verify that called builtins occur in the generated IR.

// CHECK-NOT: intel_sub_group_avc_mce_get_default_inter_base_multi_reference_penalty
// CHECK-NOT: ndrange_t
// CHECK20: ndrange_t
// CHECK: _Z16convert_char_rtec
// CHECK-NOT: _Z3ctzc
// CHECK20: _Z3ctzc
// CHECK20: _Z16convert_char_rtec
char f(char x) {
// Check functionality from OpenCL 2.0 onwards
#if defined(__OPENCL_CPP_VERSION__) || (__OPENCL_C_VERSION__ == CL_VERSION_2_0)
  ndrange_t t;
  x = ctz(x);
#endif //__OPENCL_C_VERSION__
  return convert_char_rte(x);
}

// Verify that a builtin using a write_only image3d_t type is available
// from OpenCL 2.0 onwards.

// CHECK20: _Z12write_imagef14ocl_image3d_wo
#if defined(__OPENCL_CPP_VERSION__) || (__OPENCL_C_VERSION__ >= CL_VERSION_2_0)
void test_image3dwo(write_only image3d_t img) {
  write_imagef(img, (0), (0.0f));
}
#endif //__OPENCL_C_VERSION__

// Verify that non-builtin cl_intel_planar_yuv extension is defined from
// OpenCL 1.2 onwards.
#if defined(__OPENCL_CPP_VERSION__) || (__OPENCL_C_VERSION__ >= CL_VERSION_1_2)
// expected-no-diagnostics
#else //__OPENCL_C_VERSION__
// expected-warning@+2{{unknown OpenCL extension 'cl_intel_planar_yuv' - ignoring}}
#endif //__OPENCL_C_VERSION__
#pragma OPENCL EXTENSION cl_intel_planar_yuv : enable

// CHECK-MOD: Reading modules
