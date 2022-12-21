// RUN: %clang -target aarch64-none-linux-gnu -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target arm64-none-linux-gnu -### %s -fsyntax-only 2>&1 | FileCheck %s

// CHECK: "-funwind-tables=2"

// The AArch64 PCS states that chars should be unsigned.
// CHECK: fno-signed-char

// Check for AArch64 out-of-line atomics default settings.
// RUN: %clang -target aarch64-linux-android -rtlib=compiler-rt \
// RUN: -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-OUTLINE-ATOMICS-ON %s

// RUN: %clang -target aarch64-linux-gnu -rtlib=compiler-rt \
// RUN: -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-OUTLINE-ATOMICS-ON %s

// RUN: %clang -target arm64-unknown-linux -rtlib=compiler-rt \
// RUN: -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-OUTLINE-ATOMICS-ON %s

// RUN: %clang -target aarch64--none-eabi -rtlib=compiler-rt \
// RUN: -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-OUTLINE-ATOMICS-OFF %s

// RUN: %clang -target aarch64-apple-darwin -rtlib=compiler-rt \
// RUN: -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-OUTLINE-ATOMICS-OFF %s

// RUN: %clang -target aarch64-windows-gnu -rtlib=compiler-rt \
// RUN: -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-OUTLINE-ATOMICS-OFF %s

// RUN: %clang -target aarch64-unknown-openbsd -rtlib=compiler-rt \
// RUN: -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-OUTLINE-ATOMICS-OFF %s

// RUN: %clang -target aarch64-linux-gnu -rtlib=libgcc \
// RUN: --gcc-toolchain=%S/Inputs/aarch64-linux-gnu-tree/gcc-10 \
// RUN: -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-OUTLINE-ATOMICS-ON %s

// RUN: %clang -target aarch64-linux-gnu -rtlib=libgcc \
// RUN: --gcc-toolchain=%S/Inputs/aarch64-linux-gnu-tree/gcc-7.5.0 \
// RUN: -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-OUTLINE-ATOMICS-OFF %s

// RUN: %clang -target aarch64-linux-gnu -rtlib=libgcc \
// RUN: --gcc-toolchain=%S/Inputs/aarch64-linux-gnu-tree/gcc-9.3.1 \
// RUN: -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-OUTLINE-ATOMICS-ON %s

// RUN: %clang -target aarch64-linux-gnu -rtlib=libgcc \
// RUN: --gcc-toolchain=%S/Inputs/aarch64-linux-gnu-tree/gcc-9.3.0 \
// RUN: -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-OUTLINE-ATOMICS-OFF %s

// RUN: %clang -target arm64-linux -rtlib=compiler-rt -mno-outline-atomics \
// RUN: -### -c %s 2>&1 | FileCheck \
// RUN: -check-prefixes=CHECK-OUTLINE-ATOMICS-OFF,CHECK-NO-OUTLINE-ATOMICS %s

// RUN: %clang -target aarch64-linux-gnu -rtlib=libgcc -mno-outline-atomics \
// RUN: --gcc-toolchain=%S/Inputs/aarch64-linux-gnu-tree/gcc-10 \
// RUN: -### -c %s 2>&1 | FileCheck \
// RUN: -check-prefixes=CHECK-OUTLINE-ATOMICS-OFF,CHECK-NO-OUTLINE-ATOMICS %s

// RUN: %clang -target aarch64-apple-darwin -rtlib=compiler-rt -moutline-atomics \
// RUN: -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-OUTLINE-ATOMICS-ON %s

// RUN: %clang -target aarch64-windows-gnu -rtlib=libgcc -moutline-atomics \
// RUN: --gcc-toolchain=%S/Inputs/aarch64-linux-gnu-tree/gcc-7.5.0 \
// RUN: -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-OUTLINE-ATOMICS-ON %s

// CHECK-OUTLINE-ATOMICS-ON: "-target-feature" "+outline-atomics"
// CHECK-OUTLINE-ATOMICS-OFF-NOT: "-target-feature" "+outline-atomics"
// CHECK-NO-OUTLINE-ATOMICS: "-target-feature" "-outline-atomics"
