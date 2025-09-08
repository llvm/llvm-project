// RUN: %clang -### -nogpulib -nogpuinc %s \
// RUN:   -fatomic-fine-grained-memory -fno-atomic-remote-memory -fatomic-ignore-denormal-mode \
// RUN:   2>&1 | FileCheck %s --check-prefix=CHECK-VALID

// RUN: %clang -### -nogpulib -nogpuinc %s \
// RUN:   2>&1 | FileCheck %s --check-prefix=NEG

// RUN: %clang -### -nogpulib -nogpuinc -fno-atomic-backward-compatible %s \
// RUN:   2>&1 | FileCheck %s --check-prefix=COMPAT

// CHECK-VALID: "-cc1" {{.*}}"-triple" "amdgcn-amd-amdhsa" {{.*}} "-fno-atomic-remote-memory" "-fatomic-fine-grained-memory" "-fatomic-ignore-denormal-mode"
// CHECK-VALID: "-cc1" {{.*}}"-triple" {{.*}}"-fno-atomic-remote-memory" "-fatomic-fine-grained-memory" "-fatomic-ignore-denormal-mode"

// NEG-NOT: atomic-remote-memory
// NEG-NOT: atomic-fine-grained-memory
// NEG-NOT: atomic-ignore-denormal-mode

// COMPAT: "-cc1" {{.*}}"-triple" "amdgcn-amd-amdhsa" {{.*}} "-fno-atomic-backward-compatible"
// COMPAT: "-cc1" {{.*}}"-triple" {{.*}} "-fno-atomic-backward-compatible"