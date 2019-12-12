// A basic clang -cc1 command-line, and simple environment check.

// The tests here are similar to those in riscv64-toolchain.c, however
// these tests need to create symlinks to test directory trees in order to
// set up the environment and therefore shell support is required.
// REQUIRES: shell, riscv-registered-target
// UNSUPPORTED: system-windows

// If there is no GCC install detected then the driver searches for executables
// and runtime starting from the directory tree above the driver itself.
// The test below checks that the driver correctly finds the linker and
// runtime if and only if they exist.
//
// RUN: mkdir -p %T/testroot-riscv64-baremetal-nogcc/bin
// RUN: [ ! -s %T/testroot-riscv64-baremetal-nogcc/bin/clang ] || rm %T/testroot-riscv64-baremetal-nogcc/bin/clang
// RUN: [ ! -s %T/testroot-riscv64-baremetal-nogcc/bin/riscv64-unknown-elf-ld ] || rm %T/testroot-riscv64-baremetal-nogcc/bin/riscv64-unknown-elf-ld
// RUN: [ ! -s %T/testroot-riscv64-baremetal-nogcc/riscv64-unknown-elf ] || rm %T/testroot-riscv64-baremetal-nogcc/riscv64-unknown-elf
// RUN: ln -s %clang %T/testroot-riscv64-baremetal-nogcc/bin/clang
// RUN: ln -s %S/Inputs/basic_riscv64_nogcc_tree/bin/riscv64-unknown-elf-ld %T/testroot-riscv64-baremetal-nogcc/bin/riscv64-unknown-elf-ld
// RUN: ln -s %S/Inputs/basic_riscv64_nogcc_tree/riscv64-unknown-elf %T/testroot-riscv64-baremetal-nogcc/riscv64-unknown-elf
// RUN: %T/testroot-riscv64-baremetal-nogcc/bin/clang %s -### -no-canonical-prefixes \
// RUN:    -target riscv64-unknown-elf --rtlib=platform 2>&1 \
// RUN:    | FileCheck -check-prefix=C-RV64-BAREMETAL-LP64-NOGCC %s

// C-RV64-BAREMETAL-LP64-NOGCC: "-internal-isystem" "{{.*}}Output/testroot-riscv64-baremetal-nogcc/bin/../riscv64-unknown-elf/include"
// C-RV64-BAREMETAL-LP64-NOGCC: "{{.*}}Output/testroot-riscv64-baremetal-nogcc/bin/riscv64-unknown-elf-ld"
// C-RV64-BAREMETAL-LP64-NOGCC: "{{.*}}Output/testroot-riscv64-baremetal-nogcc/bin/../riscv64-unknown-elf/lib/crt0.o"
// C-RV64-BAREMETAL-LP64-NOGCC: "{{.*}}Output/testroot-riscv64-baremetal-nogcc/lib/clang/{{[0-9.]*}}/lib/clang_rt.crtbegin-riscv64.o"
// C-RV64-BAREMETAL-LP64-NOGCC: "{{.*}}Output/testroot-riscv64-baremetal-nogcc/bin/../riscv64-unknown-elf/lib"
// C-RV64-BAREMETAL-LP64-NOGCC: "--start-group" "-lc" "-lgloss" "--end-group"
// C-RV64-BAREMETAL-LP64-NOGCC: "{{.*}}Output/testroot-riscv64-baremetal-nogcc/lib/clang/{{[0-9.]*}}/lib/libclang_rt.builtins-riscv64.a"
// C-RV64-BAREMETAL-LP64-NOGCC: "{{.*}}Output/testroot-riscv64-baremetal-nogcc/lib/clang/{{[0-9.]*}}/lib/clang_rt.crtend-riscv64.o"
