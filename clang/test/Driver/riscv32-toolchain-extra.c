// A basic clang -cc1 command-line, and simple environment check.

// The tests here are similar to those in riscv32-toolchain.c, however
// these tests need to create symlinks to test directory trees in order to
// set up the environment and therefore POSIX support is required.
// UNSUPPORTED: system-windows

// If there is no GCC install detected then the driver searches for executables
// and runtime starting from the directory tree above the driver itself.
// The test below checks that the driver correctly finds the linker and
// runtime if and only if they exist.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t/riscv32-nogcc/bin
// RUN: ln -s %clang %t/riscv32-nogcc/bin/clang
// RUN: ln -s %S/Inputs/basic_riscv32_nogcc_tree/bin/riscv32-unknown-elf-ld %t/riscv32-nogcc/bin/riscv32-unknown-elf-ld
// RUN: ln -s %S/Inputs/basic_riscv32_nogcc_tree/riscv32-unknown-elf %t/riscv32-nogcc/riscv32-unknown-elf
// RUN: %t/riscv32-nogcc/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --gcc-toolchain=%t/riscv32-nogcc/invalid \
// RUN:    --target=riscv32-unknown-elf --rtlib=platform -fuse-ld=ld 2>&1 \
// RUN:    | FileCheck -check-prefix=C-RV32-BAREMETAL-ILP32-NOGCC %s

// RUN: %t/riscv32-nogcc/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --sysroot=%t/riscv32-nogcc/bin/../riscv32-unknown-elf \
// RUN:    --target=riscv32-unknown-elf --rtlib=platform -fuse-ld=ld 2>&1 \
// RUN:    | FileCheck -check-prefix=C-RV32-BAREMETAL-ILP32-NOGCC %s

// C-RV32-BAREMETAL-ILP32-NOGCC: "-internal-isystem" "{{.*}}/riscv32-nogcc/bin/../riscv32-unknown-elf/include"
// C-RV32-BAREMETAL-ILP32-NOGCC: "{{.*}}/riscv32-nogcc/bin/riscv32-unknown-elf-ld"
// C-RV32-BAREMETAL-ILP32-NOGCC: "{{.*}}/riscv32-nogcc/bin/../riscv32-unknown-elf/lib/crt0.o"
// C-RV32-BAREMETAL-ILP32-NOGCC: "{{.*}}/riscv32-nogcc/{{.*}}/riscv32-unknown-unknown-elf/clang_rt.crtbegin.o"
// C-RV32-BAREMETAL-ILP32-NOGCC: "{{.*}}/riscv32-nogcc/bin/../riscv32-unknown-elf/lib"
// C-RV32-BAREMETAL-ILP32-NOGCC: "--start-group" "{{.*}}/riscv32-nogcc/{{.*}}/riscv32-unknown-unknown-elf/libclang_rt.builtins.a" "-lc" "-lgloss" "--end-group"
// C-RV32-BAREMETAL-ILP32-NOGCC: "{{.*}}/riscv32-nogcc/{{.*}}/riscv32-unknown-unknown-elf/clang_rt.crtend.o"
