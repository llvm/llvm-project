// Make sure -mthumb does not affect assembler triple, but -Wa,-mthumb or
// -Xassembler -mthumb does. Also check that -Wa,-mthumb or -Xassembler -mthumb
// does not affect non assembler files.

// RUN: %clang -target armv7a-linux-gnueabi -### -c -mthumb %s 2>&1 | \
// RUN: FileCheck -check-prefix=TRIPLE-ARM %s
// RUN: %clang -target armv7a-linux-gnueabi -### -c -Wa,-mthumb \
// RUN: %S/Inputs/wildcard1.c  2>&1 | FileCheck -check-prefix=TRIPLE-ARM %s

// TRIPLE-ARM: "-triple" "armv7--linux-gnueabi"

// RUN: %clang -target armv7a-linux-gnueabi -### -c -Wa,-mthumb %s 2>&1 | \
// RUN: FileCheck -check-prefix=TRIPLE-THUMB %s
// RUN: %clang -target armv7a-linux-gnueabi -### -c -Xassembler -mthumb %s \
// RUN: 2>&1 | FileCheck -check-prefix=TRIPLE-THUMB %s

// TRIPLE-THUMB: "-triple" "thumbv7--linux-gnueabi"
