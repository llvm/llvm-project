// REQUIRES: mips-registered-target
// RUN: %clang -### -c -target mips64-mti-elf -fno-PIC -mabicalls %s 2>&1 | FileCheck %s
// CHECK: warning: ignoring '-mabicalls' option as it cannot be used with non position-independent code and the N64 ABI

// RUN: %clang -### -c -target mips-mti-elf -mlong-calls %s 2>&1 | FileCheck -check-prefix=LONGCALL-IMP %s
// LONGCALL-IMP: warning: ignoring '-mlong-calls' option as it is not currently supported with the implicit usage of -mabicalls

// RUN: %clang -### -c -target mips-mti-elf -mlong-calls -mabicalls %s 2>&1 | FileCheck -check-prefix=LONGCALL-EXP %s
// LONGCALL-EXP: warning: ignoring '-mlong-calls' option as it is not currently supported with -mabicalls
