// UNSUPPORTED: system-windows
// REQUIRES: riscv-registered-target
// RUN: %clang -target riscv64be-unknown-elf -### %s 2>&1 \
// RUN:   | FileCheck %s
// RUN: %clang -target riscv64be-unknown-elf -Wno-riscv-be-experimental -### %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOWARN

// CHECK: warning: big-endian RISC-V target support is experimental
// CHECK: "-triple" "riscv64be-unknown-unknown-elf"
// NOWARN-NOT: warning: big-endian RISC-V target support is experimental
// NOWARN: "-triple" "riscv64be-unknown-unknown-elf"

/// Test dynamic linker for big-endian RISC-V Linux targets
// RUN: %clang -### %s --target=riscv64be-unknown-linux-gnu \
// RUN:   -Wno-riscv-be-experimental --rtlib=platform -mabi=lp64d 2>&1 \
// RUN:   | FileCheck -check-prefix=RV64BE-LINUX-LP64D %s
// RV64BE-LINUX-LP64D: "-dynamic-linker" "/lib/ld-linux-riscv64be-lp64d.so.1"

// RUN: %clang -### %s --target=riscv64be-unknown-linux-gnu \
// RUN:   -Wno-riscv-be-experimental --rtlib=platform -mabi=lp64 2>&1 \
// RUN:   | FileCheck -check-prefix=RV64BE-LINUX-LP64 %s
// RV64BE-LINUX-LP64: "-dynamic-linker" "/lib/ld-linux-riscv64be-lp64.so.1"

// RUN: %clang -### %s --target=riscv32be-unknown-linux-gnu \
// RUN:   -Wno-riscv-be-experimental --rtlib=platform -mabi=ilp32d 2>&1 \
// RUN:   | FileCheck -check-prefix=RV32BE-LINUX-ILP32D %s
// RV32BE-LINUX-ILP32D: "-dynamic-linker" "/lib/ld-linux-riscv32be-ilp32d.so.1"

// RUN: %clang -### %s --target=riscv32be-unknown-linux-gnu \
// RUN:   -Wno-riscv-be-experimental --rtlib=platform -mabi=ilp32 2>&1 \
// RUN:   | FileCheck -check-prefix=RV32BE-LINUX-ILP32 %s
// RV32BE-LINUX-ILP32: "-dynamic-linker" "/lib/ld-linux-riscv32be-ilp32.so.1"

/// Test big-endian RISC-V GCC multilib directory layout
// RUN: env "PATH=" %clang -### %s -fuse-ld= -no-pie \
// RUN:   --target=riscv64be-unknown-linux-gnu --rtlib=platform --unwindlib=platform -mabi=lp64 \
// RUN:   -Wno-riscv-be-experimental \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_linux_sdk_be \
// RUN:   --sysroot=%S/Inputs/multilib_riscv_linux_sdk_be/sysroot 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64BE-LINUX-MULTI-LP64 %s

// C-RV64BE-LINUX-MULTI-LP64: "{{.*}}/Inputs/multilib_riscv_linux_sdk_be/lib/gcc/riscv64be-unknown-linux-gnu/7.2.0/../../../../riscv64be-unknown-linux-gnu/bin/ld"
// C-RV64BE-LINUX-MULTI-LP64: "--sysroot={{.*}}/Inputs/multilib_riscv_linux_sdk_be/sysroot"
// C-RV64BE-LINUX-MULTI-LP64: "-m" "elf64briscv" "-X"
// C-RV64BE-LINUX-MULTI-LP64: "-dynamic-linker" "/lib/ld-linux-riscv64be-lp64.so.1"
// C-RV64BE-LINUX-MULTI-LP64: "{{.*}}/Inputs/multilib_riscv_linux_sdk_be/lib/gcc/riscv64be-unknown-linux-gnu/7.2.0/lib64/lp64/crtbegin.o"
// C-RV64BE-LINUX-MULTI-LP64: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk_be/lib/gcc/riscv64be-unknown-linux-gnu/7.2.0/lib64/lp64"
// C-RV64BE-LINUX-MULTI-LP64: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk_be/sysroot/lib64/lp64"
// C-RV64BE-LINUX-MULTI-LP64: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk_be/sysroot/usr/lib64/lp64"

// RUN: env "PATH=" %clang -### %s -fuse-ld= -no-pie \
// RUN:   --target=riscv64be-unknown-linux-gnu --rtlib=platform --unwindlib=platform -march=rv64imafd \
// RUN:   -Wno-riscv-be-experimental \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_linux_sdk_be \
// RUN:   --sysroot=%S/Inputs/multilib_riscv_linux_sdk_be/sysroot 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64BE-LINUX-MULTI-LP64D %s

// C-RV64BE-LINUX-MULTI-LP64D: "{{.*}}/Inputs/multilib_riscv_linux_sdk_be/lib/gcc/riscv64be-unknown-linux-gnu/7.2.0/../../../../riscv64be-unknown-linux-gnu/bin/ld"
// C-RV64BE-LINUX-MULTI-LP64D: "--sysroot={{.*}}/Inputs/multilib_riscv_linux_sdk_be/sysroot"
// C-RV64BE-LINUX-MULTI-LP64D: "-m" "elf64briscv"
// C-RV64BE-LINUX-MULTI-LP64D: "-dynamic-linker" "/lib/ld-linux-riscv64be-lp64d.so.1"
// C-RV64BE-LINUX-MULTI-LP64D: "{{.*}}/Inputs/multilib_riscv_linux_sdk_be/lib/gcc/riscv64be-unknown-linux-gnu/7.2.0/lib64/lp64d/crtbegin.o"
// C-RV64BE-LINUX-MULTI-LP64D: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk_be/lib/gcc/riscv64be-unknown-linux-gnu/7.2.0/lib64/lp64d"
// C-RV64BE-LINUX-MULTI-LP64D: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk_be/sysroot/lib64/lp64d"
// C-RV64BE-LINUX-MULTI-LP64D: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk_be/sysroot/usr/lib64/lp64d"

// RUN: env "PATH=" %clang -### %s -fuse-ld= -no-pie \
// RUN:   --target=riscv32be-unknown-linux-gnu --rtlib=platform --unwindlib=platform -mabi=ilp32 \
// RUN:   -Wno-riscv-be-experimental \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_linux_sdk_be \
// RUN:   --sysroot=%S/Inputs/multilib_riscv_linux_sdk_be/sysroot 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV32BE-LINUX-MULTI-ILP32 %s

// C-RV32BE-LINUX-MULTI-ILP32: "--sysroot={{.*}}/Inputs/multilib_riscv_linux_sdk_be/sysroot"
// C-RV32BE-LINUX-MULTI-ILP32: "-m" "elf32briscv" "-X"
// C-RV32BE-LINUX-MULTI-ILP32: "-dynamic-linker" "/lib/ld-linux-riscv32be-ilp32.so.1"
// C-RV32BE-LINUX-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk_be/sysroot/lib32/ilp32"
// C-RV32BE-LINUX-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk_be/sysroot/usr/lib32/ilp32"

// RUN: env "PATH=" %clang -### %s -fuse-ld= -no-pie \
// RUN:   --target=riscv32be-unknown-linux-gnu --rtlib=platform --unwindlib=platform -march=rv32imafd \
// RUN:   -Wno-riscv-be-experimental \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_linux_sdk_be \
// RUN:   --sysroot=%S/Inputs/multilib_riscv_linux_sdk_be/sysroot 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV32BE-LINUX-MULTI-ILP32D %s

// C-RV32BE-LINUX-MULTI-ILP32D: "--sysroot={{.*}}/Inputs/multilib_riscv_linux_sdk_be/sysroot"
// C-RV32BE-LINUX-MULTI-ILP32D: "-m" "elf32briscv"
// C-RV32BE-LINUX-MULTI-ILP32D: "-dynamic-linker" "/lib/ld-linux-riscv32be-ilp32d.so.1"
// C-RV32BE-LINUX-MULTI-ILP32D: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk_be/sysroot/lib32/ilp32d"
// C-RV32BE-LINUX-MULTI-ILP32D: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk_be/sysroot/usr/lib32/ilp32d"

int foo(void) {
  return 0;
}
