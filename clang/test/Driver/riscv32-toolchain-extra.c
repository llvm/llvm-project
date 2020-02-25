// A basic clang -cc1 command-line, and simple environment check.

// The tests here are similar to those in riscv32-toolchain.c, however
// these tests need to create symlinks to test directory trees in order to
// set up the environment and therefore shell support is required.
// REQUIRES: shell, riscv-registered-target
// UNSUPPORTED: system-windows

// If there is no GCC install detected then the driver searches for executables
// and runtime starting from the directory tree above the driver itself.
// The test below checks that the driver correctly finds the linker and
// runtime if and only if they exist.
//
// RUN: mkdir -p %T/testroot-riscv32-baremetal-nogcc/bin
// RUN: [ ! -s %T/testroot-riscv32-baremetal-nogcc/bin/clang ] || rm %T/testroot-riscv32-baremetal-nogcc/bin/clang
// RUN: [ ! -s %T/testroot-riscv32-baremetal-nogcc/bin/riscv32-unknown-elf-ld ] || rm %T/testroot-riscv32-baremetal-nogcc/bin/riscv32-unknown-elf-ld
// RUN: [ ! -s %T/testroot-riscv32-baremetal-nogcc/riscv32-unknown-elf ] || rm %T/testroot-riscv32-baremetal-nogcc/riscv32-unknown-elf
// RUN: ln -s %clang %T/testroot-riscv32-baremetal-nogcc/bin/clang
// RUN: ln -s %S/Inputs/basic_riscv32_nogcc_tree/bin/riscv32-unknown-elf-ld %T/testroot-riscv32-baremetal-nogcc/bin/riscv32-unknown-elf-ld
// RUN: ln -s %S/Inputs/basic_riscv32_nogcc_tree/riscv32-unknown-elf %T/testroot-riscv32-baremetal-nogcc/riscv32-unknown-elf
// RUN: %T/testroot-riscv32-baremetal-nogcc/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --gcc-toolchain=%T/testroot-riscv32-baremetal-nogcc/invalid \
// RUN:    -target riscv32-unknown-elf --rtlib=platform 2>&1 \
// RUN:    | FileCheck -check-prefix=C-RV32-BAREMETAL-ILP32-NOGCC %s

// C-RV32-BAREMETAL-ILP32-NOGCC: "-internal-isystem" "{{.*}}Output/testroot-riscv32-baremetal-nogcc/bin/../riscv32-unknown-elf/include"
// C-RV32-BAREMETAL-ILP32-NOGCC: "{{.*}}Output/testroot-riscv32-baremetal-nogcc/bin/riscv32-unknown-elf-ld"
// C-RV32-BAREMETAL-ILP32-NOGCC: "{{.*}}Output/testroot-riscv32-baremetal-nogcc/bin/../riscv32-unknown-elf/lib/crt0.o"
// C-RV32-BAREMETAL-ILP32-NOGCC: "{{.*}}Output/testroot-riscv32-baremetal-nogcc/{{.*}}/lib/clang_rt.crtbegin-riscv32.o"
// C-RV32-BAREMETAL-ILP32-NOGCC: "{{.*}}Output/testroot-riscv32-baremetal-nogcc/bin/../riscv32-unknown-elf/lib"
// C-RV32-BAREMETAL-ILP32-NOGCC: "--start-group" "-lc" "-lgloss" "--end-group"
// C-RV32-BAREMETAL-ILP32-NOGCC: "{{.*}}Output/testroot-riscv32-baremetal-nogcc/{{.*}}/lib/libclang_rt.builtins-riscv32.a"
// C-RV32-BAREMETAL-ILP32-NOGCC: "{{.*}}Output/testroot-riscv32-baremetal-nogcc/{{.*}}/lib/clang_rt.crtend-riscv32.o"
