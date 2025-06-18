// A basic clang -cc1 command-line, and simple environment check.

// The tests here are similar to those in aarch64-toolchain.c, however
// these tests need to create symlinks to test directory trees in order to
// set up the environment and therefore shell support is required.
// XFAIL: target={{.*}}-fuchsia{{.*}}
// REQUIRES: shell
// UNSUPPORTED: system-windows

// If there is no GCC install detected then the driver searches for executables
// and runtime starting from the directory tree above the driver itself.
// The test below checks that the driver correctly finds the linker and
// runtime if and only if they exist.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t/aarch64-nogcc/bin
// RUN: ln -s %clang %t/aarch64-nogcc/bin/clang
// RUN: ln -s %S/Inputs/basic_aarch64_nogcc_tree/aarch64-none-elf %t/aarch64-nogcc/aarch64-none-elf
// RUN: %t/aarch64-nogcc/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --gcc-toolchain=%t/aarch64-nogcc/invalid \
// RUN:    --target=aarch64-none-elf --rtlib=libgcc -fuse-ld=ld 2>&1 \
// RUN:    | FileCheck -check-prefix=C-ARM-BAREMETAL-NOGCC %s

// RUN: %t/aarch64-nogcc/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --sysroot=%t/aarch64-nogcc/bin/../aarch64-none-elf \
// RUN:    --target=aarch64-none-elf --rtlib=libgcc -fuse-ld=ld 2>&1 \
// RUN:    | FileCheck -check-prefix=C-ARM-BAREMETAL-NOGCC %s

// C-ARM-BAREMETAL-NOGCC: "-internal-isystem" "{{.*}}/aarch64-nogcc/bin/../aarch64-none-elf/include"
