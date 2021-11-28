/// These tests make sure that options passed to the assembler
/// via -Wa or -Xassembler are applied correctly to assembler inputs.

/// Does not apply to non assembly files
// RUN: %clang --target=aarch64-linux-gnueabi -### -c -Wa,-march=armv8.1-a \
// RUN: %S/Inputs/wildcard1.c 2>&1 | FileCheck --check-prefix=TARGET-FEATURE-1 %s
// RUN: %clang --target=aarch64-linux-gnueabi -### -c -Xassembler -march=armv8.1-a \
// RUN: %S/Inputs/wildcard1.c 2>&1 | FileCheck --check-prefix=TARGET-FEATURE-1 %s

// TARGET-FEATURE-1-NOT: "-target-feature" "+v8.1a"

/// Does apply to assembler input
// RUN: %clang --target=aarch64-linux-gnueabi -### -c -Wa,-march=armv8.2-a %s 2>&1 | \
// RUN: FileCheck --check-prefix=TARGET-FEATURE-2 %s
// RUN: %clang --target=aarch64-linux-gnueabi -### -c -Xassembler -march=armv8.2-a %s 2>&1 | \
// RUN: FileCheck --check-prefix=TARGET-FEATURE-2 %s

// TARGET-FEATURE-2: "-target-feature" "+v8.2a"

/// No unused argument warnings when there are multiple values
// RUN: %clang --target=aarch64-linux-gnueabi -### -c -Wa,-march=armv8.1-a -Wa,-march=armv8.2-a %s 2>&1 | \
// RUN: FileCheck --check-prefix=UNUSED-WARNING %s

// UNUSED-WARNING-NOT: warning: argument unused during compilation

/// Last march to assembler wins
// RUN: %clang --target=aarch64-linux-gnueabi -### -c -Wa,-march=armv8.2-a -Wa,-march=armv8.1-a %s 2>&1 | \
// RUN: FileCheck --check-prefix=MULTIPLE-VALUES %s
// RUN: %clang --target=aarch64-linux-gnueabi -### -c -Wa,-march=armv8.2-a,-march=armv8.1-a %s 2>&1 | \
// RUN: FileCheck --check-prefix=MULTIPLE-VALUES %s
// RUN: %clang --target=aarch64-linux-gnueabi -### -c -Xassembler -march=armv8.2-a -Xassembler \
// RUN: -march=armv8.1-a %s 2>&1 | FileCheck --check-prefix=MULTIPLE-VALUES %s

// MULTIPLE-VALUES: "-target-feature" "+v8.1a
// MULTIPLE-VALUES-NOT: "-target-feature" "+v8.2a

/// march to compiler and assembler, we choose the one suited to the input file type
// RUN: %clang --target=aarch64-linux-gnueabi -### -c -Wa,-march=armv8.3-a -march=armv8.4-a %s 2>&1 | \
// RUN: FileCheck --check-prefix=TARGET-FEATURE-3 %s
// RUN: %clang --target=aarch64-linux-gnueabi -### -c -Wa,-march=armv8.3-a -march=armv8.4-a \
// RUN: %S/Inputs/wildcard1.c 2>&1 | FileCheck --check-prefix=TARGET-FEATURE-4 %s

// TARGET-FEATURE-3: "-target-feature" "+v8.3a"
// TARGET-FEATURE-3-NOT: "-target-feature" "+v8.4a"
// TARGET-FEATURE-4: "-target-feature" "+v8.4a"
// TARGET-FEATURE-4-NOT: "-target-feature" "+v8.3a"

// Invalid -march settings
// RUN: %clang --target=aarch64-linux-gnueabi -### -c -Wa,-march=all %s 2>&1 | \
// RUN: FileCheck --check-prefix=INVALID-ARCH-1 %s
// RUN: %clang --target=aarch64-linux-gnueabi -### -c -Wa,-march=foobar %s 2>&1 | \
// RUN: FileCheck --check-prefix=INVALID-ARCH-2 %s

// INVALID-ARCH-1: error: the clang compiler does not support '-march=all'
// INVALID-ARCH-2: error: the clang compiler does not support '-march=foobar'
