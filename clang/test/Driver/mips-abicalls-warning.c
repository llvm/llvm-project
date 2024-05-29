// REQUIRES: mips-registered-target
// RUN: %clang -### -c --target=mips64-mti-elf -fno-pic %s 2>&1 | FileCheck -check-prefix=CHECK-PIC1-IMPLICIT %s
// CHECK-PIC1-IMPLICIT: warning: ignoring '-fno-pic' option as it cannot be used with implicit usage of -mabicalls and the N64 ABI

// RUN: %clang -### -c --target=mips64-mti-elf -fno-pic -mabicalls %s 2>&1 | FileCheck -check-prefix=CHECK-PIC1-EXPLICIT %s
// CHECK-PIC1-EXPLICIT: warning: ignoring '-fno-pic' option as it cannot be used with -mabicalls and the N64 ABI

// RUN: %clang -### -c --target=mips64-mti-elf -fno-PIC %s 2>&1 | FileCheck -check-prefix=CHECK-PIC2-IMPLICIT %s
// CHECK-PIC2-IMPLICIT: warning: ignoring '-fno-PIC' option as it cannot be used with implicit usage of -mabicalls and the N64 ABI

// RUN: %clang -### -c --target=mips64-mti-elf -fno-PIC -mabicalls %s 2>&1 | FileCheck -check-prefix=CHECK-PIC2-EXPLICIT %s
// CHECK-PIC2-EXPLICIT: warning: ignoring '-fno-PIC' option as it cannot be used with -mabicalls and the N64 ABI

// RUN: %clang -### -c --target=mips64-mti-elf -fno-pie %s 2>&1 | FileCheck -check-prefix=CHECK-PIE1-IMPLICIT %s
// CHECK-PIE1-IMPLICIT: warning: ignoring '-fno-pie' option as it cannot be used with implicit usage of -mabicalls and the N64 ABI

// RUN: %clang -### -c --target=mips64-mti-elf -fno-pie -mabicalls %s 2>&1 | FileCheck -check-prefix=CHECK-PIE1-EXPLICIT %s
// CHECK-PIE1-EXPLICIT: warning: ignoring '-fno-pie' option as it cannot be used with -mabicalls and the N64 ABI

// RUN: %clang -### -c --target=mips64-mti-elf -fno-PIE %s 2>&1 | FileCheck -check-prefix=CHECK-PIE2-IMPLICIT %s
// CHECK-PIE2-IMPLICIT: warning: ignoring '-fno-PIE' option as it cannot be used with implicit usage of -mabicalls and the N64 ABI

// RUN: %clang -### -c --target=mips64-mti-elf -fno-PIE -mabicalls %s 2>&1 | FileCheck -check-prefix=CHECK-PIE2-EXPLICIT %s
// CHECK-PIE2-EXPLICIT: warning: ignoring '-fno-PIE' option as it cannot be used with -mabicalls and the N64 ABI

// RUN: %clang -### -c --target=mips-mti-elf -mlong-calls %s 2>&1 | FileCheck -check-prefix=LONGCALL-IMP %s
// LONGCALL-IMP: warning: ignoring '-mlong-calls' option as it is not currently supported with the implicit usage of -mabicalls

// RUN: %clang -### -c --target=mips-mti-elf -mlong-calls -mabicalls %s 2>&1 | FileCheck -check-prefix=LONGCALL-EXP %s
// LONGCALL-EXP: warning: ignoring '-mlong-calls' option as it is not currently supported with -mabicalls
