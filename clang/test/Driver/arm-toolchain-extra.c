// A basic clang -cc1 command-line, and simple environment check.

// The tests here are similar to those in arm-toolchain.c, however
// these tests need to create symlinks to test directory trees in order to
// set up the environment and therefore shell support is required.
// UNSUPPORTED: system-windows

// If there is no GCC install detected then the driver searches for executables
// and runtime starting from the directory tree above the driver itself.
// The test below checks that the driver correctly finds the linker and
// runtime if and only if they exist.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t/arm-nogcc/bin
// RUN: ln -s %clang %t/arm-nogcc/bin/clang
// RUN: ln -s %S/Inputs/basic_arm_nogcc_tree/armv6m-none-eabi %t/arm-nogcc/armv6m-none-eabi
// RUN: ln -s %S/Inputs/basic_arm_nogcc_tree/bin/armv6m-none-eabi-ld %t/arm-nogcc/bin/armv6m-none-eabi-ld
// RUN: %t/arm-nogcc/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --gcc-toolchain=%t/arm-nogcc/invalid \
// RUN:    --target=armv6m-none-eabi --rtlib=libgcc --unwindlib=platform -fuse-ld=ld 2>&1 \
// RUN:    | FileCheck -check-prefix=C-ARM-BAREMETAL-NOGCC %s

// RUN: %t/arm-nogcc/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --sysroot=%t/arm-nogcc/bin/../armv6m-none-eabi \
// RUN:    --target=armv6m-none-eabi --rtlib=libgcc --unwindlib=platform -fuse-ld=ld 2>&1 \
// RUN:    | FileCheck -check-prefix=C-ARM-BAREMETAL-NOGCC %s

// C-ARM-BAREMETAL-NOGCC: "-internal-isystem" "{{.*}}/arm-nogcc/bin/../armv6m-none-eabi/include"
// C-ARM-BAREMETAL-NOGCC: "{{.*}}/arm-nogcc/bin/armv6m-none-eabi-ld"
// C-ARM-BAREMETAL-NOGCC: "{{.*}}/arm-nogcc/bin/../armv6m-none-eabi/lib/crt0.o"
// C-ARM-BAREMETAL-NOGCC: "{{.*}}/arm-nogcc/{{.*}}/armv6m-none-eabi/lib/crtbegin.o"
// C-ARM-BAREMETAL-NOGCC: "{{.*}}/arm-nogcc/bin/../armv6m-none-eabi/lib"
// C-ARM-BAREMETAL-NOGCC: "{{.*}}.o" "--start-group" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "-lc" "-lgloss" "--end-group"
// C-ARM-BAREMETAL-NOGCC: "{{.*}}/arm-nogcc/{{.*}}/armv6m-none-eabi/lib/crtend.o"

