// RUN: %clang -### -nogpulib -nogpuinc %s \
// RUN:   -fatomic-fine-grained-memory -fno-atomic-remote-memory -fatomic-ignore-denormal-mode \
// RUN:   2>&1 | FileCheck %s --check-prefix=CHECK-VALID

// CHECK-VALID: "-cc1" {{.*}}"-triple" "amdgcn-amd-amdhsa" {{.*}}"-fatomic-fine-grained-memory" "-fatomic-ignore-denormal-mode"
// CHECK-VALID: "-cc1" {{.*}}"-triple" {{.*}}"-fatomic-fine-grained-memory" "-fatomic-ignore-denormal-mode"
