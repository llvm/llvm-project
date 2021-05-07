// REQUIRES: shell
// UNSUPPORTED: system-windows

// Test that --sysroot always comes after any library paths.

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target armv6m-none-eabi -fuse-ld=lld \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/sys-root -L%S/lib \
// RUN:   | FileCheck --check-prefix=CHECK-V6M -DLIB_DIR=%S/lib %s
// CHECK-V6M: "{{.*}}clang{{.*}}" "-cc1" "-triple" "thumbv6m-none-unknown-eabi"
// CHECK-V6M-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-V6M-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-V6M-NEXT: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-V6M-SAME: "-L[[LIB_DIR]]"
// CHECK-V6M-SAME: "-L[[SYSROOT]]{{/|\\\\}}lib"
// CHECK-V6M-SAME: "-L[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}baremetal"
// CHECK-V6M-SAME: "-lc" "-lm" "-lclang_rt.builtins-armv6m"
// CHECK-V6M-SAME: "-o" "{{.*}}.o"

// Test that when a --sysroot is not provided, driver picks the default
// location correctly if available.

// RUN: rm -rf %T/baremetal_default_sysroot
// RUN: mkdir -p %T/baremetal_default_sysroot/bin
// RUN: mkdir -p %T/baremetal_default_sysroot/lib/clang-runtimes/armv6m-none-eabi
// RUN: ln -s %clang %T/baremetal_default_sysroot/bin/clang

// RUN: %T/baremetal_default_sysroot/bin/clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target armv6m-none-eabi \
// RUN:   | FileCheck --check-prefix=CHECK-V6M-DEFAULT-SYSROOT %s
// CHECK-V6M-DEFAULT-SYSROOT: "{{.*}}clang{{.*}}" "-cc1" "-triple" "thumbv6m-none-unknown-eabi"
// CHECK-V6M-DEFAULT-SYSROOT-SAME: "-internal-isystem" "{{.*}}/baremetal_default_sysroot{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}armv6m-none-eabi{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECk-V6M-DEFAULT-SYSROOT-SAME: "-internal-isystem" "{{.*}}/baremetal_default_sysroot{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}armv6m-none-eabi{{[/\\]+}}include"
// CHECK-V6M-DEFAULT-SYSROOT-SAME: "-x" "c++" "{{.*}}baremetal-sysroot.cpp"
// CHECK-V6M-DEFAULT-SYSROOT-NEXT: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-V6M-DEFAULT-SYSROOT-SAME: "-L{{.*}}/baremetal_default_sysroot{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}armv6m-none-eabi{{[/\\]+}}lib"
// CHECK-V6M-DEFAULT-SYSROOT-SAME: "-lc" "-lm" "-lclang_rt.builtins-armv6m"
// CHECK-V6M-DEFAULT-SYSROOT-SAME: "-o" "{{.*}}.o"
