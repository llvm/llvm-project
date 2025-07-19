// RUN: %clang -### --target=armv7-unknown-none-eabi -mcpu=cortex-m4 --sysroot= -fuse-ld=ld %s 2>&1 | FileCheck --check-prefix=NOLTO %s
// NOLTO: {{".*ld.*"}} {{.*}}
// NOLTO-NOT: "-plugin-opt=mcpu"

// RUN: %clang -### --target=armv7-unknown-none-eabi -mcpu=cortex-m4 --sysroot= -fuse-ld=ld -flto -O3 %s 2>&1 | FileCheck --check-prefix=LTO %s
// LTO: {{".*ld.*"}} {{.*}} "-plugin-opt=mcpu=cortex-m4" "-plugin-opt=O3"

// Ensure that, for freestanding -none targets, the linker does not call into gcc.
// We do this by checking if clang is trying to pass "-fuse-ld=bfd" to the linker command.
// RUN: %clang --target=aarch64-unknown-none-elf -ccc-print-bindings %s 2>&1 | FileCheck --check-prefix=LDAARCH64 %s
// LDAARCH64: "baremetal::Linker"
// RUN: %clang --target=loongarch64-unknown-none-elf -ccc-print-bindings %s 2>&1 | FileCheck --check-prefix=LDLOONGARCH64 %s
// LDLOONGARCH64: "GNU::Linker"
// RUN: %clang --target=riscv64-unknown-none-elf -ccc-print-bindings %s 2>&1 | FileCheck --check-prefix=LDRISCV64 %s
// LDRISCV64: "baremetal::Linker"
// RUN: %clang --target=x86_64-unknown-none-elf -ccc-print-bindings %s 2>&1 | FileCheck --check-prefix=LDX8664 %s
// LDX8664: "GNU::Linker"
// RUN: %clang --target=i386-unknown-none-elf -ccc-print-bindings %s 2>&1 | FileCheck --check-prefix=LDI386 %s
// LDI386: "GNU::Linker"
