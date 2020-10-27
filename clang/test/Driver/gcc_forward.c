// RUN: %clang -### %s -target aarch64-none-elf \
// RUN:   --coverage -e _start -fuse-ld=lld --ld-path=ld -nostdlib -r -rdynamic -static -static-pie \
// RUN:   2>&1 | FileCheck --check-prefix=FORWARD %s
// FORWARD: gcc{{[^"]*}}" "--coverage" "-fuse-ld=lld" "--ld-path=ld" "-nostdlib" "-rdynamic" "-static" "-static-pie" "-o" "a.out" "{{.*}}.o" "-e" "_start" "-r"

// Check that we don't try to forward -Xclang or -mlinker-version to GCC.
// PR12920 -- Check also we may not forward W_Group options to GCC.
//
// RUN: %clang -target powerpc-unknown-unknown \
// RUN:   %s \
// RUN:   -Wall -Wdocumentation \
// RUN:   -Xclang foo-bar \
// RUN:   -pie -march=x86-64 \
// RUN:   -mlinker-version=10 -### 2> %t
// RUN: FileCheck < %t %s
//
// clang -cc1
// CHECK: clang
// CHECK: "-Wall" "-Wdocumentation"
// CHECK: "-o" "{{[^"]+}}.o"
//
// gcc as ld.
// CHECK: gcc{{[^"]*}}" "-pie"
// CHECK-NOT: "-mlinker-version=10"
// CHECK-NOT: "-Xclang"
// CHECK-NOT: "foo-bar"
// CHECK-NOT: "-Wall"
// CHECK-NOT: "-Wdocumentation"
// CHECK-NOT: -march
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
