// XFAIL: *
// RUN: %clang -S -### -cl-std=CL %s | FileCheck --check-prefix=CHECK-CL %s
// RUN: %clang -S -### -cl-std=CL1.1 %s | FileCheck --check-prefix=CHECK-CL11 %s
// RUN: %clang -S -### -cl-std=CL1.2 %s | FileCheck --check-prefix=CHECK-CL12 %s
// RUN: %clang -S -### -cl-std=CL2.0 %s | FileCheck --check-prefix=CHECK-CL20 %s
// RUN: %clang -S -### -cl-opt-disable %s | FileCheck --check-prefix=CHECK-OPT-DISABLE %s
// RUN: %clang -S -### -cl-strict-aliasing %s | FileCheck --check-prefix=CHECK-STRICT-ALIASING %s
// RUN: %clang -S -### -cl-std=CL1.1 -cl-strict-aliasing %s | FileCheck --check-prefix=CHECK-INVALID-OPENCL-VERSION11 %s
// RUN: %clang -S -### -cl-std=CL1.2 -cl-strict-aliasing %s | FileCheck --check-prefix=CHECK-INVALID-OPENCL-VERSION12 %s
// RUN: %clang -S -### -cl-std=CL2.0 -cl-strict-aliasing %s | FileCheck --check-prefix=CHECK-INVALID-OPENCL-VERSION20 %s
// RUN: %clang -S -### -cl-single-precision-constant %s | FileCheck --check-prefix=CHECK-SINGLE-PRECISION-CONST %s
// RUN: %clang -S -### -cl-finite-math-only %s | FileCheck --check-prefix=CHECK-FINITE-MATH-ONLY %s
// RUN: %clang -S -### -cl-kernel-arg-info %s | FileCheck --check-prefix=CHECK-KERNEL-ARG-INFO %s
// RUN: %clang -S -### -cl-unsafe-math-optimizations %s | FileCheck --check-prefix=CHECK-UNSAFE-MATH-OPT %s
// RUN: %clang -S -### -cl-fast-relaxed-math %s | FileCheck --check-prefix=CHECK-FAST-RELAXED-MATH %s
// RUN: %clang -S -### -cl-mad-enable %s | FileCheck --check-prefix=CHECK-MAD-ENABLE %s
// RUN: %clang -S -### -cl-denorms-are-zero %s | FileCheck --check-prefix=CHECK-DENORMS-ARE-ZERO %s
// RUN: not %clang_cc1 -cl-std=c99 -DOPENCL %s 2>&1 | FileCheck --check-prefix=CHECK-C99 %s
// RUN: not %clang_cc1 -cl-std=invalid -DOPENCL %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID %s

// CHECK-CL: .*clang.* "-cc1" .* "-cl-std=CL"
// CHECK-CL11: .*clang.* "-cc1" .* "-cl-std=CL1.1"
// CHECK-CL12: .*clang.* "-cc1" .* "-cl-std=CL1.2"
// CHECK-CL20: .*clang.* "-cc1" .* "-cl-std=CL2.0"
// CHECK-OPT-DISABLE: .*clang.* "-cc1" .* "-cl-opt-disable"
// CHECK-STRICT-ALIASING: .*clang.* "-cc1" .* "-cl-strict-aliasing"
// CHECK-INVALID-OPENCL-VERSION11: OpenCL version 1.1 does not support the option 'cl-strict-aliasing'
// CHECK-INVALID-OPENCL-VERSION12: OpenCL version 1.2 does not support the option 'cl-strict-aliasing'
// CHECK-INVALID-OPENCL-VERSION20: OpenCL version 2.0 does not support the option 'cl-strict-aliasing'
// CHECK-SINGLE-PRECISION-CONST: .*clang.* "-cc1" .* "-cl-single-precision-constant"
// CHECK-FINITE-MATH-ONLY: .*clang.* "-cc1" .* "-cl-finite-math-only"
// CHECK-KERNEL-ARG-INFO: .*clang.* "-cc1" .* "-cl-kernel-arg-info"
// CHECK-UNSAFE-MATH-OPT: .*clang.* "-cc1" .* "-cl-unsafe-math-optimizations"
// CHECK-FAST-RELAXED-MATH: .*clang.* "-cc1" .* "-cl-fast-relaxed-math"
// CHECK-MAD-ENABLE: .*clang.* "-cc1" .* "-cl-mad-enable"
// CHECK-DENORMS-ARE-ZERO: .*clang.* "-cc1" .* "-cl-denorms-are-zero"
// CHECK-C99: error: invalid argument '-cl-std=c99' not allowed with 'OpenCL'
// CHECK-INVALID: error: invalid value 'invalid' in '-cl-std=invalid'

kernel void func(void);
