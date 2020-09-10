// REQUIRES: clang-driver, x86-registered-target, amdgpu-registered-target

// RUN: %clang -### -target x86_64-unknown-linux-gnu --offload-arch=gfx906 \
// RUN:   -fsanitize=address \
// RUN:   -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck %s

// CHECK-NOT: {{"[^"]*clang[^"]*".* "-fcuda-is-device".* "-fsanitize=address"}}
// CHECK: {{"[^"]*clang[^"]*".* "-triple" "x86_64-unknown-linux-gnu".* "-fsanitize=address"}}
