// RUN: %clang -### --target=m68k-unknown-linux -mcpu=68000 %s 2>&1 | FileCheck --check-prefix=CHECK-M00 %s
// RUN: %clang -### --target=m68k-unknown-linux -mcpu=m68000 %s 2>&1 | FileCheck --check-prefix=CHECK-M00 %s
// RUN: %clang -### --target=m68k-unknown-linux -mcpu=M68000 %s 2>&1 | FileCheck --check-prefix=CHECK-M00 %s
// RUN: %clang -### --target=m68k-unknown-linux -m68000 %s 2>&1 | FileCheck --check-prefix=CHECK-M00 %s
// CHECK-M00: "-target-cpu" "M68000"

// RUN: %clang -### --target=m68k-unknown-linux -mcpu=68010 %s 2>&1 | FileCheck --check-prefix=CHECK-M10 %s
// RUN: %clang -### --target=m68k-unknown-linux -mcpu=m68010 %s 2>&1 | FileCheck --check-prefix=CHECK-M10 %s
// RUN: %clang -### --target=m68k-unknown-linux -mcpu=M68010 %s 2>&1 | FileCheck --check-prefix=CHECK-M10 %s
// RUN: %clang -### --target=m68k-unknown-linux -m68010 %s 2>&1 | FileCheck --check-prefix=CHECK-M10 %s
// CHECK-M10: "-target-cpu" "M68010"

// RUN: %clang -### --target=m68k-unknown-linux -mcpu=68020 %s 2>&1 | FileCheck --check-prefix=CHECK-M20 %s
// RUN: %clang -### --target=m68k-unknown-linux -mcpu=m68020 %s 2>&1 | FileCheck --check-prefix=CHECK-M20 %s
// RUN: %clang -### --target=m68k-unknown-linux -mcpu=M68020 %s 2>&1 | FileCheck --check-prefix=CHECK-M20 %s
// RUN: %clang -### --target=m68k-unknown-linux -m68020 %s 2>&1 | FileCheck --check-prefix=CHECK-M20 %s
// CHECK-M20: "-target-cpu" "M68020"

// RUN: %clang -### --target=m68k-unknown-linux -mcpu=68030 %s 2>&1 | FileCheck --check-prefix=CHECK-M30 %s
// RUN: %clang -### --target=m68k-unknown-linux -mcpu=m68030 %s 2>&1 | FileCheck --check-prefix=CHECK-M30 %s
// RUN: %clang -### --target=m68k-unknown-linux -mcpu=M68030 %s 2>&1 | FileCheck --check-prefix=CHECK-M30 %s
// RUN: %clang -### --target=m68k-unknown-linux -m68030 %s 2>&1 | FileCheck --check-prefix=CHECK-M30 %s
// CHECK-M30: "-target-cpu" "M68030"

// RUN: %clang -### --target=m68k-unknown-linux -mcpu=68040 %s 2>&1 | FileCheck --check-prefix=CHECK-M40 %s
// RUN: %clang -### --target=m68k-unknown-linux -mcpu=m68040 %s 2>&1 | FileCheck --check-prefix=CHECK-M40 %s
// RUN: %clang -### --target=m68k-unknown-linux -mcpu=M68040 %s 2>&1 | FileCheck --check-prefix=CHECK-M40 %s
// RUN: %clang -### --target=m68k-unknown-linux -m68040 %s 2>&1 | FileCheck --check-prefix=CHECK-M40 %s
// CHECK-M40: "-target-cpu" "M68040"

// RUN: %clang -### --target=m68k-unknown-linux -mcpu=68060 %s 2>&1 | FileCheck --check-prefix=CHECK-M60 %s
// RUN: %clang -### --target=m68k-unknown-linux -mcpu=m68060 %s 2>&1 | FileCheck --check-prefix=CHECK-M60 %s
// RUN: %clang -### --target=m68k-unknown-linux -mcpu=M68060 %s 2>&1 | FileCheck --check-prefix=CHECK-M60 %s
// RUN: %clang -### --target=m68k-unknown-linux -m68060 %s 2>&1 | FileCheck --check-prefix=CHECK-M60 %s
// CHECK-M60: "-target-cpu" "M68060"
