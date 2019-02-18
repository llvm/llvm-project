// RUN: %clang -### -c -mcmodel=tiny %s 2>&1 | FileCheck -check-prefix CHECK-TINY %s
// RUN: %clang -### -c -mcmodel=small %s 2>&1 | FileCheck -check-prefix CHECK-SMALL %s
// RUN: %clang -### -S -mcmodel=kernel %s 2>&1 | FileCheck -check-prefix CHECK-KERNEL %s
// RUN: %clang -### -c -mcmodel=medium %s 2>&1 | FileCheck -check-prefix CHECK-MEDIUM %s
// RUN: %clang -### -S -mcmodel=large %s 2>&1 | FileCheck -check-prefix CHECK-LARGE %s
// RUN: not %clang -c -mcmodel=lager %s 2>&1 | FileCheck -check-prefix CHECK-INVALID %s

// CHECK-TINY: "-mcode-model" "tiny"
// CHECK-SMALL: "-mcode-model" "small"
// CHECK-KERNEL: "-mcode-model" "kernel"
// CHECK-MEDIUM: "-mcode-model" "medium"
// CHECK-LARGE: "-mcode-model" "large"

// CHECK-INVALID: error: invalid value 'lager' in '-mcode-model lager'

