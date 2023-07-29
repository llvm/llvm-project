// RUN: %clang -target thumb-none-windows-eabi-coff -mcpu=cortex-m7 -### -c %s 2>&1 \
// RUN: | FileCheck %s --check-prefix CHECK-V7
// CHECK-V7-NOT: error: the target architecture 'thumbv7em' is not supported by the target 'thumbv7em-none-windows-eabihf'

// RUN: %clang -target thumb-none-windows-eabi-coff -mcpu=cortex-m1 -### -c %s 2>&1 \
// RUN: | FileCheck %s --check-prefix CHECK-V6
// CHECK-V6: error: the target architecture 'thumbv6m' is not supported by the target 'thumbv6m-none-windows-eabihf'

