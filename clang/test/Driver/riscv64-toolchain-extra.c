// A basic clang -cc1 command-line, and simple environment check.

// The tests here are similar to those in riscv64-toolchain.c, however
// these tests need to create symlinks to test directory trees in order to
// set up the environment and therefore shell support is required.
// REQUIRES: shell
// UNSUPPORTED: system-windows

// If there is no GCC install detected then the driver searches for executables
// and runtime starting from the directory tree above the driver itself.
// The test below checks that the driver correctly finds the linker and
// runtime if and only if they exist.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t/riscv64-nogcc/bin
// RUN: ln -s %clang %t/riscv64-nogcc/bin/clang
// RUN: ln -s %S/Inputs/basic_riscv64_nogcc_tree/bin/riscv64-unknown-elf-ld %t/riscv64-nogcc/bin/riscv64-unknown-elf-ld
// RUN: ln -s %S/Inputs/basic_riscv64_nogcc_tree/riscv64-unknown-elf %t/riscv64-nogcc/riscv64-unknown-elf
// RUN: %t/riscv64-nogcc/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --gcc-toolchain=%t/riscv64-nogcc/invalid \
// RUN:    --target=riscv64-unknown-elf --rtlib=platform -fuse-ld= 2>&1 \
// RUN:    | FileCheck -check-prefix=C-RV64-BAREMETAL-LP64-NOGCC %s

// RUN: %t/riscv64-nogcc/bin/clang %s -### -no-canonical-prefixes \
// RUN:    --sysroot=%t/riscv64-nogcc/bin/../riscv64-unknown-elf \
// RUN:    --target=riscv64-unknown-elf --rtlib=platform -fuse-ld= 2>&1 \
// RUN:    | FileCheck -check-prefix=C-RV64-BAREMETAL-LP64-NOGCC %s

// C-RV64-BAREMETAL-LP64-NOGCC: "-internal-isystem" "{{.*}}/riscv64-nogcc/bin/../riscv64-unknown-elf/include"
// C-RV64-BAREMETAL-LP64-NOGCC: "{{.*}}/riscv64-nogcc/bin/riscv64-unknown-elf-ld"
// C-RV64-BAREMETAL-LP64-NOGCC: "{{.*}}/riscv64-nogcc/bin/../riscv64-unknown-elf/lib/crt0.o"
// C-RV64-BAREMETAL-LP64-NOGCC: "{{.*}}/riscv64-nogcc/{{.*}}/riscv64-unknown-unknown-elf/clang_rt.crtbegin.o"
// C-RV64-BAREMETAL-LP64-NOGCC: "{{.*}}/riscv64-nogcc/bin/../riscv64-unknown-elf/lib"
// C-RV64-BAREMETAL-LP64-NOGCC: "--start-group" "-lc" "-lgloss" "--end-group"
// C-RV64-BAREMETAL-LP64-NOGCC: "{{.*}}/riscv64-nogcc/{{.*}}/riscv64-unknown-unknown-elf/libclang_rt.builtins.a"
// C-RV64-BAREMETAL-LP64-NOGCC: "{{.*}}/riscv64-nogcc/{{.*}}/riscv64-unknown-unknown-elf/clang_rt.crtend.o"
