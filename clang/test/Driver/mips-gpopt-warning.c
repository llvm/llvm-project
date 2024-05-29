// REQUIRES: mips-registered-target
// RUN: %clang -### -c --target=mips-mti-elf %s -mgpopt 2>&1 | FileCheck -check-prefix=IMPLICIT %s
// IMPLICIT: warning: ignoring '-mgpopt' option as it cannot be used with the implicit usage of -mabicalls

// RUN: %clang -### -c --target=mips-mti-elf %s -mgpopt -mabicalls 2>&1 | FileCheck -check-prefix=EXPLICIT %s
// EXPLICIT: warning: ignoring '-mgpopt' option as it cannot be used with -mabicalls
