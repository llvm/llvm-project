// RUN: %clang --target=aarch64-none-elf -march=armv8a+rdm -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-RDM %s
// RUN: %clang --target=aarch64-none-elf -march=armv8a+rdma -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-RDM %s
// RUN: %clang --target=aarch64-none-elf -mcpu=generic+rdm -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-RDM %s
// RUN: %clang --target=aarch64-none-elf -mcpu=falkor -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-RDM %s
// RUN: %clang --target=aarch64-none-elf -mcpu=thunderx2t99 -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-RDM %s
// CHECK-RDM: "-target-feature" "+rdm"

// RUN: %clang --target=aarch64-none-elf -march=armv8a+nordm -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-NORDM %s
// RUN: %clang --target=aarch64-none-elf -march=armv8a+nordma -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-NORDM %s
// RUN: %clang --target=aarch64-none-elf -mcpu=generic+nordm -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-NORDM %s
// CHECK-NORDM-NOT: "-target-feature" "+rdm"
//
// RUN: %clang --target=aarch64-none-elf -march=armv8.1a -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-RDM %s
// RUN: %clang --target=aarch64-none-elf -march=armv8.1a+nordm -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-NORDM-DEFAULT %s
// RUN: %clang --target=aarch64-none-elf -march=armv8.1a+nordma -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-NORDM-DEFAULT %s
// CHECK-NORDM-DEFAULT: "-target-feature" "-rdm"
