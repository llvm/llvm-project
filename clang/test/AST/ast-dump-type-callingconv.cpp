// Verify there is a space after the parens when priting callingconv attributes.
// RUN: %clang_cc1 -DDEVICE -triple spirv64 -ast-dump -ast-dump-filter foo %s \
// RUN: | FileCheck -check-prefix=CHECK-DEVICE --strict-whitespace %s

// RUN: %clang_cc1 -DVECTOR -triple aarch64 -ast-dump -ast-dump-filter foo %s \
// RUN: | FileCheck -check-prefix=CHECK-VECTOR --strict-whitespace %s

// RUN: %clang_cc1 -DSVE -triple aarch64 -ast-dump -ast-dump-filter foo %s \
// RUN: | FileCheck -check-prefix=CHECK-SVE --strict-whitespace %s

#ifdef DEVICE
// CHECK-DEVICE-NOT: ()__attribute__((device_kernel))
void foo() __attribute__((device_kernel));
#endif

#ifdef VECTOR
// CHECK-VECTOR-NOT: ()__attribute__((aarch64_vector_pcs))
void foo()  __attribute__((aarch64_vector_pcs));
#endif

#ifdef SVE
// CHECK-SVE-NOT: ()__attribute__((aarch64_sve_pcs))
void foo()  __attribute__((aarch64_sve_pcs));
#endif
