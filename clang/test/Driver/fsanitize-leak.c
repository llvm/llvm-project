// RUN: %clang --target=x86_64-linux-gnu -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL
// CHECK-SANL: "-fsanitize=leak"

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=address,leak -fno-sanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANA-SANL-NO-SANA
// CHECK-SANA-SANL-NO-SANA: "-fsanitize=leak"

// RUN: %clang --target=i686-linux-gnu -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-X86
// CHECK-SANL-X86: "-fsanitize=leak"

// RUN: %clang --target=i686-linux-gnu -fsanitize=address,leak -fno-sanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANA-SANL-NO-SANA-X86
// CHECK-SANA-SANL-NO-SANA-X86: "-fsanitize=leak"

// RUN: %clang --target=arm-linux-gnu -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-ARM
// CHECK-SANL-ARM: "-fsanitize=leak"

// RUN: %clang --target=arm-linux-gnu -fsanitize=address,leak -fno-sanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANA-SANL-NO-SANA-ARM
// CHECK-SANA-SANL-NO-SANA-ARM: "-fsanitize=leak"

// RUN: %clang --target=thumb-linux -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-THUMB
// CHECK-SANL-THUMB: "-fsanitize=leak"

// RUN: %clang --target=thumb-linux -fsanitize=address,leak -fno-sanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANA-SANL-NO-SANA-THUMB
// CHECK-SANA-SANL-NO-SANA-THUMB: "-fsanitize=leak"

// RUN: %clang --target=armeb-linux-gnu -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-ARMEB
// CHECK-SANL-ARMEB: "-fsanitize=leak"

// RUN: %clang --target=armeb-linux-gnu -fsanitize=address,leak -fno-sanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANA-SANL-NO-SANA-ARMEB
// CHECK-SANA-SANL-NO-SANA-ARMEB: "-fsanitize=leak"

// RUN: %clang --target=thumbeb-linux -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-THUMBEB
// CHECK-SANL-THUMBEB: "-fsanitize=leak"

// RUN: %clang --target=thumbeb-linux -fsanitize=address,leak -fno-sanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANA-SANL-NO-SANA-THUMBEB
// CHECK-SANA-SANL-NO-SANA-THUMBEB: "-fsanitize=leak"

// RUN: not %clang --target=mips-unknown-linux -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-MIPS
// CHECK-SANL-MIPS: unsupported option '-fsanitize=leak' for target 'mips-unknown-linux'

// RUN: not %clang --target=mips-unknown-freebsd -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-MIPS-FREEBSD
// CHECK-SANL-MIPS-FREEBSD: unsupported option '-fsanitize=leak' for target 'mips-unknown-freebsd'

// RUN: %clang --target=mips64-unknown-freebsd -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-MIPS64-FREEBSD
// CHECK-SANL-MIPS64-FREEBSD: "-fsanitize=leak"

// RUN: %clang --target=powerpc64-unknown-linux -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-PPC64
// RUN: %clang --target=powerpc64le-unknown-linux -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-PPC64
// CHECK-SANL-PPC64: "-fsanitize=leak"
// RUN: not %clang --target=powerpc-unknown-linux -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-PPC
// CHECK-SANL-PPC: unsupported option '-fsanitize=leak' for target 'powerpc-unknown-linux'

// RUN: %clang --target=riscv64-linux-gnu -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-RISCV64
// CHECK-SANL-RISCV64: "-fsanitize=leak"

// RUN: %clang --target=riscv64-linux-gnu -fsanitize=address,leak -fno-sanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANA-SANL-NO-SANA-RISCV64
// CHECK-SANA-SANL-NO-SANA-RISCV64: "-fsanitize=leak"

// RUN: %clang --target=loongarch64-unknown-linux-gnu -fsanitize=leak %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANL-LOONGARCH64
// CHECK-SANL-LOONGARCH64: "-fsanitize=leak"

// RUN: %clang --target=loongarch64-unknown-linux-gnu -fsanitize=address,leak -fno-sanitize=address %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANA-SANL-NO-SANA-LOONGARCH64
// CHECK-SANA-SANL-NO-SANA-LOONGARCH64: "-fsanitize=leak"
