// RUN: %clang -### -nogpulib -nogpuinc %s \
// RUN:   -fatomic=no_fine_grained_memory:off,no_remote_memory:on,ignore_denormal_mode:on \
// RUN:   2>&1 | FileCheck %s --check-prefix=CHECK-VALID

// CHECK-VALID: "-cc1" {{.*}}"-triple" "amdgcn-amd-amdhsa" {{.*}}"-fatomic=no_fine_grained_memory:off,no_remote_memory:on,ignore_denormal_mode:on"
// CHECK-VALID: "-cc1" {{.*}}"-triple" {{.*}}"-fatomic=no_fine_grained_memory:off,no_remote_memory:on,ignore_denormal_mode:on"

// RUN: not %clang -### -nogpulib -nogpuinc %s \
// RUN:   -fatomic=invalid_key:on 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID-KEY

// CHECK-INVALID-KEY: invalid argument 'invalid_key:on' to '-fatomic='; key 'invalid_key' is not allowed; allowed keys are 'no_fine_grained_memory', 'no_remote_memory', 'ignore_denormal_mode'

// RUN: not %clang -### -nogpulib -nogpuinc %s \
// RUN:   -fatomic=no_fine_grained_memory:invalid 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID-VALUE

// CHECK-INVALID-VALUE: invalid argument 'no_fine_grained_memory:invalid' to '-fatomic='; value 'invalid' is invalid; must be boolean (true/false, 1/0, yes/no, on/off)

// RUN: not %clang -### -nogpulib -nogpuinc %s \
// RUN:   -fatomic=no_fine_grained_memory 2>&1 | FileCheck %s --check-prefix=CHECK-MISSING-VALUE

// CHECK-MISSING-VALUE: invalid argument 'no_fine_grained_memory' to '-fatomic='; must be a comma-separated list of key:value pairs

// RUN: not %clang -### -nogpulib -nogpuinc %s \
// RUN:   -fatomic=no_fine_grained_memory:on,no_fine_grained_memory:off 2>&1 | FileCheck %s --check-prefix=CHECK-DUPLICATE-KEY

// CHECK-DUPLICATE-KEY: invalid argument 'no_fine_grained_memory:off' to '-fatomic='; duplicate key 'no_fine_grained_memory'

// RUN: %clang -### -nogpulib -nogpuinc %s \
// RUN:   -fatomic= 2>&1 | FileCheck %s --check-prefix=CHECK-EMPTY

// CHECK-EMPTY: clang: warning: joined argument expects additional value: '-fatomic=' [-Wunused-command-line-argument]
