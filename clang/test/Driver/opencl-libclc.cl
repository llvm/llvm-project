// RUN: %clang -### -target amdgcn-amd-amdhsa --no-offloadlib --libclc-lib=:%S/Inputs/libclc/libclc.bc %s 2>&1 | FileCheck %s
// RUN: %clang -### -target amdgcn-amd-amdhsa --no-offloadlib --libclc-lib=:%S/Inputs/libclc/subdir/libclc.bc %s 2>&1 | FileCheck %s --check-prefix CHECK-SUBDIR

// RUN: %clang -### -target amdgcn-amd-amdhsa -mcpu=gfx908 --no-offloadlib -resource-dir=%S/Inputs/resource_dir --libclc-lib %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-INFER
// RUN: %clang -### -target amdgcn-mesa-mesa3d -mcpu=gfx908 --no-offloadlib -resource-dir=%S/Inputs/resource_dir --libclc-lib %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-INFER-MESA3D

// RUN: not %clang -### -target amdgcn-amd-amdhsa --no-offloadlib --libclc-lib=:%S/Inputs/libclc/subdir/not-here.bc %s 2>&1 | FileCheck %s --check-prefix CHECK-ERROR

// CHECK: -mlink-builtin-bitcode{{.*}}Inputs{{/|\\\\}}libclc{{/|\\\\}}libclc.bc
// CHECK-SUBDIR: -mlink-builtin-bitcode{{.*}}Inputs{{/|\\\\}}libclc{{/|\\\\}}subdir{{/|\\\\}}libclc.bc

// CHECK-INFER: -mlink-builtin-bitcode{{.*}}Inputs{{/|\\\\}}resource_dir{{/|\\\\}}lib{{/|\\\\}}libclc{{/|\\\\}}gfx908-amdgcn--.bc
// CHECK-INFER-MESA3D: -mlink-builtin-bitcode{{.*}}Inputs{{/|\\\\}}resource_dir{{/|\\\\}}lib{{/|\\\\}}libclc{{/|\\\\}}gfx908-amdgcn-mesa-mesa3d.bc

// CHECK-ERROR: no libclc library{{.*}}not-here.bc' found in the clang resource directory
