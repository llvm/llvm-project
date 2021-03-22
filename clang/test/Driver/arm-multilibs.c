// RUN: %clang -target armv7-linux-gnueabi --sysroot=%S/Inputs/multilib_arm_linux_tree -### -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ARM %s
// RUN: %clang -target thumbv7-linux-gnueabi --sysroot=%S/Inputs/multilib_arm_linux_tree -### -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ARM %s

// RUN: %clang -target armv7-linux-gnueabihf --sysroot=%S/Inputs/multilib_armhf_linux_tree -### -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ARMHF %s
// RUN: %clang -target thumbv7-linux-gnueabihf --sysroot=%S/Inputs/multilib_armhf_linux_tree -### -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ARMHF %s

// RUN: %clang -target armv7eb-linux-gnueabi --sysroot=%S/Inputs/multilib_armeb_linux_tree -### -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ARMEB %s
// RUN: %clang -target thumbv7eb-linux-gnueabi --sysroot=%S/Inputs/multilib_armeb_linux_tree -### -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ARMEB %s

// RUN: %clang -target armv7eb-linux-gnueabihf --sysroot=%S/Inputs/multilib_armebhf_linux_tree -### -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ARMEBHF %s
// RUN: %clang -target thumbv7eb-linux-gnueabihf --sysroot=%S/Inputs/multilib_armebhf_linux_tree -### -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-ARMEBHF %s

// CHECK-ARM: "-internal-externc-isystem" "{{.*}}/usr/include/arm-linux-gnueabi"
// CHECK-ARMHF: "-internal-externc-isystem" "{{.*}}/usr/include/arm-linux-gnueabihf"
// CHECK-ARMEB: "-internal-externc-isystem" "{{.*}}/usr/include/armeb-linux-gnueabi"
// CHECK-ARMEBHF: "-internal-externc-isystem" "{{.*}}/usr/include/armeb-linux-gnueabihf"

