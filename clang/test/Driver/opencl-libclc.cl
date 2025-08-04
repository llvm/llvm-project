// RUN: %clang -### -target amdgcn-amd-amdhsa --no-offloadlib --libclc-lib=:%S/Inputs/libclc/libclc.bc %s 2>&1 | FileCheck %s
// RUN: %clang -### -target amdgcn-amd-amdhsa --no-offloadlib --libclc-lib=:%S/Inputs/libclc/subdir/libclc.bc %s 2>&1 | FileCheck %s --check-prefix CHECK-SUBDIR

// RUN: not %clang -### -target amdgcn-amd-amdhsa --no-offloadlib --libclc-lib=:%S/Inputs/libclc/subdir/not-here.bc %s 2>&1 | FileCheck %s --check-prefix CHECK-ERROR

// CHECK: -mlink-builtin-bitcode{{.*}}Inputs{{/|\\\\}}libclc{{/|\\\\}}libclc.bc
// CHECK-SUBDIR: -mlink-builtin-bitcode{{.*}}Inputs{{/|\\\\}}libclc{{/|\\\\}}subdir{{/|\\\\}}libclc.bc

// CHECK-ERROR: no libclc library{{.*}}not-here.bc' found in the clang resource directory
