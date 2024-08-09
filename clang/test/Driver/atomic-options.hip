// RUN: %clang -### -nogpulib -nogpuinc %s \
// RUN:   -fatomic=no_fine_grained_memory:off,no_remote_memory:on,ignore_denormal_mode:on \
// RUN:   2>&1 | FileCheck %s --check-prefix=CHECK-VALID

// CHECK-VALID: "-cc1" {{.*}}"-triple" "amdgcn-amd-amdhsa" {{.*}}"-fatomic=no_fine_grained_memory:off,no_remote_memory:on,ignore_denormal_mode:on"
// CHECK-VALID: "-cc1" {{.*}}"-triple" {{.*}}"-fatomic=no_fine_grained_memory:off,no_remote_memory:on,ignore_denormal_mode:on"

// RUN: not %clang -### -nogpulib -nogpuinc %s \
// RUN:   -fatomic=invalid_key:on 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID-KEY

// CHECK-INVALID-KEY: clang: error: invalid argument 'invalid_key:on' to -fatomic=; must be a comma-separated list of key:value pairs, where allowed keys are 'no_fine_grained_memory', 'no_remote_memory', 'ignore_denormal_mode', and values are 'on' or 'off', and each key must be unique

// RUN: not %clang -### -nogpulib -nogpuinc %s \
// RUN:   -fatomic=no_fine_grained_memory:invalid 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID-VALUE

// CHECK-INVALID-VALUE: clang: error: invalid argument 'no_fine_grained_memory:invalid' to -fatomic=; must be a comma-separated list of key:value pairs, where allowed keys are 'no_fine_grained_memory', 'no_remote_memory', 'ignore_denormal_mode', and values are 'on' or 'off', and each key must be unique

// RUN: not %clang -### -nogpulib -nogpuinc %s \
// RUN:   -fatomic=no_fine_grained_memory 2>&1 | FileCheck %s --check-prefix=CHECK-MISSING-VALUE

// CHECK-MISSING-VALUE: clang: error: invalid argument 'no_fine_grained_memory' to -fatomic=; must be a comma-separated list of key:value pairs, where allowed keys are 'no_fine_grained_memory', 'no_remote_memory', 'ignore_denormal_mode', and values are 'on' or 'off', and each key must be unique

// RUN: not %clang -### -nogpulib -nogpuinc %s \
// RUN:   -fatomic=no_fine_grained_memory:on,no_fine_grained_memory:off 2>&1 | FileCheck %s --check-prefix=CHECK-DUPLICATE-KEY

// CHECK-DUPLICATE-KEY: clang: error: invalid argument 'no_fine_grained_memory:off' to -fatomic=; must be a comma-separated list of key:value pairs, where allowed keys are 'no_fine_grained_memory', 'no_remote_memory', 'ignore_denormal_mode', and values are 'on' or 'off', and each key must be unique

// RUN: %clang -### -nogpulib -nogpuinc %s \
// RUN:   -fatomic= 2>&1 | FileCheck %s --check-prefix=CHECK-EMPTY

// CHECK-EMPTY: clang: warning: joined argument expects additional value: '-fatomic=' [-Wunused-command-line-argument]
