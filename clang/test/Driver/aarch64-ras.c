// RAS is off by default for v8a, but can be enabled by +ras (this is not architecturally valid)
// RUN: %clang --target=aarch64-none-elf -march=armv8a+ras -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-RAS %s
// RUN: %clang --target=aarch64-none-elf -march=armv8-a+ras -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-RAS %s
// RUN: %clang --target=aarch64-none-elf -mcpu=generic+ras -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-RAS %s
// RUN: %clang --target=aarch64-none-elf -mcpu=cortex-a75 -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-RAS %s
// RUN: %clang --target=aarch64-none-elf -mcpu=cortex-a55 -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-RAS %s
// CHECK-RAS: "-target-feature" "+ras"

// RUN: %clang --target=aarch64-none-elf -march=armv8a+noras -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-NORAS %s
// RUN: %clang --target=aarch64-none-elf -mcpu=generic+noras -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-NORAS %s
// CHECK-NORAS: "-target-feature" "-ras"

// RAS is on by default for v8.2a, but can be disabled by +noras
// FIXME: in the current implementation, RAS is not on by default at all for v8.2a (the test says it all...)
// RUN: %clang --target=aarch64 -march=armv8.2a  -### -c %s 2>&1 | FileCheck -check-prefix=V82ARAS %s
// RUN: %clang --target=aarch64 -march=armv8.2-a -### -c %s 2>&1 | FileCheck -check-prefix=V82ARAS %s
// V82ARAS-NOT: "-target-feature" "+ras"
// V82ARAS-NOT: "-target-feature" "-ras"
// RUN: %clang --target=aarch64 -march=armv8.2a+noras  -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NORAS %s
// RUN: %clang --target=aarch64 -march=armv8.2-a+noras -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NORAS %s
