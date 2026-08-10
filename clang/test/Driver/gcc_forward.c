// RUN: %clang -### %s -target x86-none-elf \
// RUN:   --coverage -e _start -fuse-ld=lld --ld-path=ld -nostartfiles \
// RUN:   -nostdlib -r -rdynamic -specs=nosys.specs -static -static-pie \
// RUN:   2>&1 | FileCheck --check-prefix=FORWARD %s
// FORWARD: gcc{{[^"]*}}" "--coverage" "-fuse-ld=lld" "--ld-path=ld" "-nostartfiles" "-nostdlib" "-rdynamic" "-specs=nosys.specs" "-static" "-static-pie" "-o" "a.out" "{{.*}}.o" "-e" "_start" "-r"

// Check that we don't try to forward -Xclang or -mlinker-version to GCC.
// PR12920 -- Check also we may not forward W_Group options to GCC.
//
// RUN: not %clang --target=powerpc-unknown-unknown \
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
