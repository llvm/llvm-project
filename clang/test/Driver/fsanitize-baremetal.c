// * Test BareMetal toolchain sanitizer support *

// RUN: %clang --target=arm-arm-non-eabi -fsanitize=address %s -### 2>&1 | FileCheck %s -check-prefix=ADDRESS-BAREMETAL
// RUN: %clang --target=aarch64-none-elf -fsanitize=address %s -### 2>&1 | FileCheck %s -check-prefix=ADDRESS-BAREMETAL
// ADDRESS-BAREMETAL: "-fsanitize=address"

// RUN: %clang --target=arm-arm-none-eabi -fsanitize=kernel-address %s -### 2>&1 | FileCheck %s -check-prefix=KERNEL-ADDRESS-BAREMETAL
// RUN: %clang --target=aarch64-none-elf -fsanitize=kernel-address %s -### 2>&1 | FileCheck %s -check-prefix=KERNEL-ADDRESS-BAREMETAL
// KERNEL-ADDRESS-BAREMETAL: "-fsanitize=kernel-address"

// RUN: %clang --target=aarch64-none-elf -fsanitize=hwaddress %s -### 2>&1 | FileCheck %s -check-prefix=HWADDRESS-BAREMETAL
// HWADDRESS-BAREMETAL: "-fsanitize=hwaddress"

// RUN: %clang --target=aarch64-none-elf -fsanitize=kernel-hwaddress %s -### 2>&1 | FileCheck %s -check-prefix=KERNEL-HWADDRESS-BAREMETAL
// KERNEL-HWADDRESS-BAREMETAL: "-fsanitize=kernel-hwaddress"

// RUN: %clang --target=aarch64-none-elf -fsanitize=thread %s -### 2>&1 | FileCheck %s -check-prefix=THREAD-BAREMETAL
// THREAD-BAREMETAL: "-fsanitize=thread"

// RUN: %clang --target=arm-arm-none-eabi -fsanitize=vptr %s -### 2>&1 | FileCheck %s -check-prefix=VPTR-BAREMETAL
// RUN: %clang --target=aarch64-none-elf -fsanitize=vptr %s -### 2>&1 | FileCheck %s -check-prefix=VPTR-BAREMETAL
// VPTR-BAREMETAL: "-fsanitize=vptr"

// RUN: %clang --target=arm-arm-none-eabi -fsanitize=safe-stack %s -### 2>&1 | FileCheck %s -check-prefix=SAFESTACK-BAREMETAL
// RUN: %clang --target=aarch64-none-elf -fsanitize=safe-stack %s -### 2>&1 | FileCheck %s -check-prefix=SAFESTACK-BAREMETAL
// SAFESTACK-BAREMETAL: "-fsanitize=safe-stack"

// RUN: %clang --target=arm-arm-none-eabi -fsanitize=undefined %s -### 2>&1 | FileCheck %s -check-prefix=UNDEFINED-BAREMETAL
// RUN: %clang --target=aarch64-none-elf -fsanitize=undefined %s -### 2>&1 | FileCheck %s -check-prefix=UNDEFINED-BAREMETAL
// UNDEFINED-BAREMETAL: "-fsanitize={{.*}}

// RUN: %clang --target=arm-arm-none-eabi -fsanitize=scudo %s -### 2>&1 | FileCheck %s -check-prefix=SCUDO-BAREMETAL
// RUN: %clang --target=aarch64-none-elf -fsanitize=scudo %s -### 2>&1 | FileCheck %s -check-prefix=SCUDO-BAREMETAL
// SCUDO-BAREMETAL: "-fsanitize=scudo"

// RUN: %clang --target=arm-arm-none-eabi -fsanitize=function %s -### 2>&1 | FileCheck %s -check-prefix=FUNCTION-BAREMETAL
// RUN: %clang --target=aarch64-none-elf -fsanitize=function %s -### 2>&1 | FileCheck %s -check-prefix=FUNCTION-BAREMETAL
// FUNCTION-BAREMETAL: "-fsanitize=function"

// RUN: not %clang --target=aarch64-none-elf -fsanitize=memory %s -### 2>&1 | FileCheck %s -check-prefix=UNSUPPORTED-BAREMETAL
// RUN: not %clang --target=aarch64-none-elf -fsanitize=kernel-memory %s -### 2>&1 | FileCheck %s -check-prefix=UNSUPPORTED-BAREMETAL
// RUN: not %clang --target=aarch64-none-elf -fsanitize=leak %s -### 2>&1 | FileCheck %s -check-prefix=UNSUPPORTED-BAREMETAL
// RUN: not %clang --target=aarch64-none-elf -fsanitize=dataflow %s -### 2>&1 | FileCheck %s -check-prefix=UNSUPPORTED-BAREMETAL
// RUN: not %clang --target=arm-arm-none-eabi -fsanitize=shadow-call-stack %s -### 2>&1 | FileCheck %s -check-prefix=UNSUPPORTED-BAREMETAL
// UNSUPPORTED-BAREMETAL: unsupported option '-fsanitize={{.*}}' for target
