// Check that we don't try to forward -Xclang or -mlinker-version to GCC.
// PR12920 -- Check also we may not forward W_Group options to GCC.
//
// RUN: %clang -target powerpc-unknown-unknown \
// RUN:   %s \
// RUN:   -Wall -Wdocumentation \
// RUN:   -Xclang foo-bar \
// RUN:   -march=x86-64 \
// RUN:   -mlinker-version=10 -### 2> %t
// RUN: FileCheck < %t %s
//
// clang -cc1
// CHECK: clang
// CHECK: "-Wall" "-Wdocumentation"
// CHECK: "-o" "{{[^"]+}}.o"
//
// gcc as ld.
// CHECK: gcc{{[^"]*}}"
// CHECK-NOT: "-mlinker-version=10"
// CHECK-NOT: "-Xclang"
// CHECK-NOT: "foo-bar"
// CHECK-NOT: "-Wall"
// CHECK-NOT: "-Wdocumentation"
// CHECK: -march
// CHECK-NOT: "-mlinker-version=10"
// CHECK-NOT: "-Xclang"
// CHECK-NOT: "foo-bar"
// CHECK-NOT: "-Wall"
// CHECK-NOT: "-Wdocumentation"
// CHECK: "-o" "a.out"

// Check that we're not forwarding -g options to the assembler
// RUN: %clang -g -target x86_64-unknown-linux-gnu -no-integrated-as -c %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ASM %s
// CHECK-ASM: as
// CHECK-ASM-NOT: "-g"

// Check that we're not forwarding -mno-unaligned-access.
// RUN: %clang -target aarch64-none-elf -mno-unaligned-access %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ARM %s
// CHECK-ARM: gcc{{[^"]*}}"
// CHECK-ARM-NOT: -mno-unaligned-access
