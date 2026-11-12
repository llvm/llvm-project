// RUN: %clang -target x86_64-apple-darwin10 \
// RUN:   -### -fsyntax-only -fasm-blocks %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-BLOCKS < %t %s

// RUN: %clang -target x86_64-apple-darwin10 \
// RUN:   -### -fsyntax-only -fno-asm-blocks -fasm-blocks %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-BLOCKS < %t %s

// CHECK-BLOCKS: "-fasm-blocks"

// RUN: %clang -target x86_64-apple-darwin10 \
// RUN:   -### -fsyntax-only -fasm-blocks -fno-asm-blocks %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-BLOCKS < %t %s

// CHECK-NO-BLOCKS-NOT: "-fasm-blocks"

// RUN: %clang -target x86_64-apple-darwin10 \
// RUN:   -### -fsyntax-only -fno-gnu-inline-asm %s 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-NO-GNU-INLINE-ASM %s
// RUN: %clang -target x86_64-apple-darwin10 \
// RUN:   -### -fsyntax-only -fgnu-inline-asm -fno-gnu-inline-asm %s 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-NO-GNU-INLINE-ASM %s

// CHECK-NO-GNU-INLINE-ASM: "-fno-gnu-inline-asm"
