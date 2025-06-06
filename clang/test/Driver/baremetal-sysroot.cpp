// REQUIRES: shell
// UNSUPPORTED: system-windows

// Test that when a --sysroot is not provided, driver picks the default
// location correctly if available.

// RUN: rm -rf %T/baremetal_default_sysroot
// RUN: mkdir -p %T/baremetal_default_sysroot/bin
// RUN: mkdir -p %T/baremetal_default_sysroot/lib/clang-runtimes/armv6m-none-eabi
// RUN: ln -s %clang %T/baremetal_default_sysroot/bin/clang

// RUN: %T/baremetal_default_sysroot/bin/clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target armv6m-none-eabi --sysroot= \
// RUN:   | FileCheck --check-prefix=CHECK-V6M-C %s
// CHECK-V6M-C: "{{.*}}clang{{.*}}" "-cc1" "-triple" "thumbv6m-unknown-none-eabi"
// CHECK-V6M-C-SAME: "-internal-isystem" "{{.*}}/baremetal_default_sysroot{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}armv6m-none-eabi{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-V6M-C-SAME: "-internal-isystem" "{{.*}}/baremetal_default_sysroot{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}armv6m-none-eabi{{[/\\]+}}include"
// CHECK-V6M-C-SAME: "-x" "c++" "{{.*}}baremetal-sysroot.cpp"
// CHECK-V6M-C-NEXT: "{{[^"]*}}ld{{(\.(lld|bfd|gold))?}}{{(\.exe)?}}" "{{.*}}.o" "-Bstatic"
// CHECK-V6M-C-SAME: "-L{{.*}}/baremetal_default_sysroot{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}armv6m-none-eabi{{[/\\]+}}lib"
// CHECK-V6M-C-SAME: "{{[^"]*}}libclang_rt.builtins.a"
// CHECK-V6M-C-SAME: "-lc"
// CHECK-V6M-C-SAME: "-o" "{{.*}}.o"
