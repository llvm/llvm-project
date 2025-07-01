// RUN: %clang -### -target amdgcn-amd-amdhsa --no-offloadlib --libclc-lib=:%S/Inputs/libclc/libclc.bc %s 2>&1 | FileCheck %s
// RUN: env LIBRARY_PATH=%S/Inputs/libclc %clang -### -target amdgcn-amd-amdhsa --no-offloadlib --libclc-lib=libclc %s 2>&1 | FileCheck %s
// RUN: env LIBRARY_PATH=%S/Inputs/libclc %clang -### -target amdgcn-amd-amdhsa --no-offloadlib --libclc-lib=:libclc.bc %s 2>&1 | FileCheck %s

// RUN: env LIBRARY_PATH=%S/Inputs/libclc/subdir %clang -### -target amdgcn-amd-amdhsa --no-offloadlib --libclc-lib=libclc %s 2>&1 | FileCheck %s --check-prefix CHECK-SUBDIR
// RUN: env LIBRARY_PATH=%S/Inputs/libclc/subdir %clang -### -target amdgcn-amd-amdhsa --no-offloadlib --libclc-lib=:libclc.bc %s 2>&1 | FileCheck %s --check-prefix CHECK-SUBDIR
// RUN: env LIBRARY_PATH=%S/Inputs/libclc %clang -### -target amdgcn-amd-amdhsa --no-offloadlib --libclc-lib=:subdir/libclc.bc %s 2>&1 | FileCheck %s --check-prefix CHECK-SUBDIR

// CHECK: -mlink-builtin-bitcode{{.*}}Inputs{{/|\\\\}}libclc{{/|\\\\}}libclc.bc
// CHECK-SUBDIR: -mlink-builtin-bitcode{{.*}}Inputs{{/|\\\\}}libclc{{/|\\\\}}subdir{{/|\\\\}}libclc.bc
