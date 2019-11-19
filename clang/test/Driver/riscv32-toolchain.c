// A basic clang -cc1 command-line, and simple environment check.

// RUN: %clang %s -### -no-canonical-prefixes -target riscv32 2>&1 | FileCheck -check-prefix=CC1 %s
// CC1: clang{{.*}} "-cc1" "-triple" "riscv32"

// RUN: %clang %s -### -no-canonical-prefixes \
// RUN:   -target riscv32-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv32_tree \
// RUN:   --sysroot=%S/Inputs/basic_riscv32_tree/riscv32-unknown-elf 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV32-BAREMETAL-ILP32 %s

// C-RV32-BAREMETAL-ILP32: "-fuse-init-array"
// C-RV32-BAREMETAL-ILP32: "{{.*}}Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv32-unknown-elf-ld"
// C-RV32-BAREMETAL-ILP32: "--sysroot={{.*}}/Inputs/basic_riscv32_tree/riscv32-unknown-elf"
// C-RV32-BAREMETAL-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/riscv32-unknown-elf/lib{{/|\\\\}}crt0.o"
// C-RV32-BAREMETAL-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1{{/|\\\\}}crtbegin.o"
// C-RV32-BAREMETAL-ILP32: "-L{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1"
// C-RV32-BAREMETAL-ILP32: "-L{{.*}}/Inputs/basic_riscv32_tree/riscv32-unknown-elf/lib"
// C-RV32-BAREMETAL-ILP32: "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// C-RV32-BAREMETAL-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1{{/|\\\\}}crtend.o"

// RUN: %clang %s -### -no-canonical-prefixes \
// RUN:   -target riscv32-unknown-elf \
// RUN:   --sysroot= \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv32_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV32-BAREMETAL-NOSYSROOT-ILP32 %s

// C-RV32-BAREMETAL-NOSYSROOT-ILP32: "-fuse-init-array"
// C-RV32-BAREMETAL-NOSYSROOT-ILP32: "{{.*}}Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv32-unknown-elf-ld"
// C-RV32-BAREMETAL-NOSYSROOT-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1/../../..{{/|\\\\}}..{{/|\\\\}}riscv32-unknown-elf/lib{{/|\\\\}}crt0.o"
// C-RV32-BAREMETAL-NOSYSROOT-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1{{/|\\\\}}crtbegin.o"
// C-RV32-BAREMETAL-NOSYSROOT-ILP32: "-L{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1"
// C-RV32-BAREMETAL-NOSYSROOT-ILP32: "-L{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1/../../..{{/|\\\\}}..{{/|\\\\}}riscv32-unknown-elf{{/|\\\\}}lib"
// C-RV32-BAREMETAL-NOSYSROOT-ILP32: "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// C-RV32-BAREMETAL-NOSYSROOT-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1{{/|\\\\}}crtend.o"

// RUN: %clangxx %s -### -no-canonical-prefixes \
// RUN:   -target riscv32-unknown-elf -stdlib=libstdc++ \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv32_tree \
// RUN:   --sysroot=%S/Inputs/basic_riscv32_tree/riscv32-unknown-elf 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-RV32-BAREMETAL-ILP32 %s

// CXX-RV32-BAREMETAL-ILP32: "-fuse-init-array"
// CXX-RV32-BAREMETAL-ILP32: "-internal-isystem" "{{.*}}Inputs/basic_riscv32_tree/riscv32-unknown-elf/include/c++{{/|\\\\}}8.0.1"
// CXX-RV32-BAREMETAL-ILP32: "{{.*}}Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv32-unknown-elf-ld"
// CXX-RV32-BAREMETAL-ILP32: "--sysroot={{.*}}/Inputs/basic_riscv32_tree/riscv32-unknown-elf"
// CXX-RV32-BAREMETAL-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/riscv32-unknown-elf/lib{{/|\\\\}}crt0.o"
// CXX-RV32-BAREMETAL-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1{{/|\\\\}}crtbegin.o"
// CXX-RV32-BAREMETAL-ILP32: "-L{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1"
// CXX-RV32-BAREMETAL-ILP32: "-L{{.*}}/Inputs/basic_riscv32_tree/riscv32-unknown-elf/lib"
// CXX-RV32-BAREMETAL-ILP32: "-lstdc++" "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// CXX-RV32-BAREMETAL-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1{{/|\\\\}}crtend.o"

// RUN: %clangxx %s -### -no-canonical-prefixes \
// RUN:   -target riscv32-unknown-elf -stdlib=libstdc++ \
// RUN:   --sysroot= \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv32_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-RV32-BAREMETAL-NOSYSROOT-ILP32 %s

// CXX-RV32-BAREMETAL-NOSYSROOT-ILP32: "-fuse-init-array"
// CXX-RV32-BAREMETAL-NOSYSROOT-ILP32: "-internal-isystem" "{{.*}}Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1/../../..{{/|\\\\}}..{{/|\\\\}}riscv32-unknown-elf/include/c++{{/|\\\\}}8.0.1"
// CXX-RV32-BAREMETAL-NOSYSROOT-ILP32: "{{.*}}Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv32-unknown-elf-ld"
// CXX-RV32-BAREMETAL-NOSYSROOT-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1/../../..{{/|\\\\}}..{{/|\\\\}}riscv32-unknown-elf/lib{{/|\\\\}}crt0.o"
// CXX-RV32-BAREMETAL-NOSYSROOT-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1{{/|\\\\}}crtbegin.o"
// CXX-RV32-BAREMETAL-NOSYSROOT-ILP32: "-L{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1"
// CXX-RV32-BAREMETAL-NOSYSROOT-ILP32: "-L{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1/../../..{{/|\\\\}}..{{/|\\\\}}riscv32-unknown-elf/lib"
// CXX-RV32-BAREMETAL-NOSYSROOT-ILP32: "-lstdc++" "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// CXX-RV32-BAREMETAL-NOSYSROOT-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1{{/|\\\\}}crtend.o"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld=ld \
// RUN:   -target riscv32-unknown-linux-gnu -mabi=ilp32 \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_linux_sdk \
// RUN:   --sysroot=%S/Inputs/multilib_riscv_linux_sdk/sysroot 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV32-LINUX-MULTI-ILP32 %s

// C-RV32-LINUX-MULTI-ILP32: "-fuse-init-array"
// C-RV32-LINUX-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-linux-gnu/bin{{/|\\\\}}ld"
// C-RV32-LINUX-MULTI-ILP32: "--sysroot={{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot"
// C-RV32-LINUX-MULTI-ILP32: "-m" "elf32lriscv"
// C-RV32-LINUX-MULTI-ILP32: "-dynamic-linker" "/lib/ld-linux-riscv32-ilp32.so.1"
// C-RV32-LINUX-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib32/ilp32{{/|\\\\}}crtbegin.o"
// C-RV32-LINUX-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib32/ilp32"
// C-RV32-LINUX-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/lib32/ilp32"
// C-RV32-LINUX-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/usr/lib32/ilp32"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld=ld \
// RUN:   -target riscv32-unknown-linux-gnu -march=rv32imafd \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_linux_sdk \
// RUN:   --sysroot=%S/Inputs/multilib_riscv_linux_sdk/sysroot 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV32-LINUX-MULTI-ILP32D %s

// C-RV32-LINUX-MULTI-ILP32D: "-fuse-init-array"
// C-RV32-LINUX-MULTI-ILP32D: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-linux-gnu/bin{{/|\\\\}}ld"
// C-RV32-LINUX-MULTI-ILP32D: "--sysroot={{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot"
// C-RV32-LINUX-MULTI-ILP32D: "-m" "elf32lriscv"
// C-RV32-LINUX-MULTI-ILP32D: "-dynamic-linker" "/lib/ld-linux-riscv32-ilp32d.so.1"
// C-RV32-LINUX-MULTI-ILP32D: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib32/ilp32d{{/|\\\\}}crtbegin.o"
// C-RV32-LINUX-MULTI-ILP32D: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib32/ilp32d"
// C-RV32-LINUX-MULTI-ILP32D: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/lib32/ilp32d"
// C-RV32-LINUX-MULTI-ILP32D: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/usr/lib32/ilp32d"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld=ld \
// RUN:   -target riscv32-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV32I-BAREMETAL-MULTI-ILP32 %s

// C-RV32I-BAREMETAL-MULTI-ILP32: "-fuse-init-array"
// C-RV32I-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/bin{{/|\\\\}}ld"
// C-RV32I-BAREMETAL-MULTI-ILP32: "-m" "elf32lriscv"
// C-RV32I-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/lib/rv32imac/ilp32{{/|\\\\}}crt0.o"
// C-RV32I-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/rv32imac/ilp32{{/|\\\\}}crtbegin.o"
// C-RV32I-BAREMETAL-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0"
// C-RV32I-BAREMETAL-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/lib"
// C-RV32I-BAREMETAL-MULTI-ILP32: "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// C-RV32I-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/rv32imac/ilp32{{/|\\\\}}crtend.o"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld=ld \
// RUN:   -target riscv32-unknown-elf -march=rv32im -mabi=ilp32\
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV32IM-BAREMETAL-MULTI-ILP32 %s

// C-RV32IM-BAREMETAL-MULTI-ILP32: "-fuse-init-array"
// C-RV32IM-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/bin{{/|\\\\}}ld"
// C-RV32IM-BAREMETAL-MULTI-ILP32: "-m" "elf32lriscv"
// C-RV32IM-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/lib/rv32im/ilp32{{/|\\\\}}crt0.o"
// C-RV32IM-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/rv32im/ilp32{{/|\\\\}}crtbegin.o"
// C-RV32IM-BAREMETAL-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0"
// C-RV32IM-BAREMETAL-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/lib"
// C-RV32IM-BAREMETAL-MULTI-ILP32: "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// C-RV32IM-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/rv32im/ilp32{{/|\\\\}}crtend.o"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld=ld \
// RUN:   -target riscv32-unknown-elf -march=rv32iac -mabi=ilp32\
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV32IAC-BAREMETAL-MULTI-ILP32 %s

// C-RV32IAC-BAREMETAL-MULTI-ILP32: "-fuse-init-array"
// C-RV32IAC-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/bin{{/|\\\\}}ld"
// C-RV32IAC-BAREMETAL-MULTI-ILP32: "-m" "elf32lriscv"
// C-RV32IAC-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/lib/rv32iac/ilp32{{/|\\\\}}crt0.o"
// C-RV32IAC-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/rv32iac/ilp32{{/|\\\\}}crtbegin.o"
// C-RV32IAC-BAREMETAL-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0"
// C-RV32IAC-BAREMETAL-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/lib"
// C-RV32IAC-BAREMETAL-MULTI-ILP32: "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// C-RV32IAC-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/rv32iac/ilp32{{/|\\\\}}crtend.o"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld=ld \
// RUN:   -target riscv32-unknown-elf -march=rv32imac -mabi=ilp32\
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV32IMAC-BAREMETAL-MULTI-ILP32 %s

// C-RV32IMAC-BAREMETAL-MULTI-ILP32: "-fuse-init-array"
// C-RV32IMAC-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/bin{{/|\\\\}}ld"
// C-RV32IMAC-BAREMETAL-MULTI-ILP32: "-m" "elf32lriscv"
// C-RV32IMAC-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/lib/rv32imac/ilp32{{/|\\\\}}crt0.o"
// C-RV32IMAC-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/rv32imac/ilp32{{/|\\\\}}crtbegin.o"
// C-RV32IMAC-BAREMETAL-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0"
// C-RV32IMAC-BAREMETAL-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/lib"
// C-RV32IMAC-BAREMETAL-MULTI-ILP32: "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// C-RV32IMAC-BAREMETAL-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/rv32imac/ilp32{{/|\\\\}}crtend.o"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld=ld \
// RUN:   -target riscv32-unknown-elf -march=rv32imafc -mabi=ilp32f \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV32IMAFC-BAREMETAL-MULTI-ILP32F %s

// C-RV32IMAFC-BAREMETAL-MULTI-ILP32F: "-fuse-init-array"
// C-RV32IMAFC-BAREMETAL-MULTI-ILP32F: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/bin{{/|\\\\}}ld"
// C-RV32IMAFC-BAREMETAL-MULTI-ILP32F: "-m" "elf32lriscv"
// C-RV32IMAFC-BAREMETAL-MULTI-ILP32F: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/lib/rv32imafc/ilp32f{{/|\\\\}}crt0.o"
// C-RV32IMAFC-BAREMETAL-MULTI-ILP32F: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/rv32imafc/ilp32f{{/|\\\\}}crtbegin.o"
// C-RV32IMAFC-BAREMETAL-MULTI-ILP32F: "-L{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0"
// C-RV32IMAFC-BAREMETAL-MULTI-ILP32F: "-L{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/lib"
// C-RV32IMAFC-BAREMETAL-MULTI-ILP32F: "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// C-RV32IMAFC-BAREMETAL-MULTI-ILP32F: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/rv32imafc/ilp32f{{/|\\\\}}crtend.o"

// RUN: %clang -target riscv32 %s -emit-llvm -S -o - | FileCheck %s

typedef __builtin_va_list va_list;
typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef __WCHAR_TYPE__ wchar_t;
typedef __WINT_TYPE__ wint_t;


// Check Alignments

// CHECK: @align_c = dso_local global i32 1
int align_c = __alignof(char);

// CHECK: @align_s = dso_local global i32 2
int align_s = __alignof(short);

// CHECK: @align_i = dso_local global i32 4
int align_i = __alignof(int);

// CHECK: @align_wc = dso_local global i32 4
int align_wc = __alignof(wchar_t);

// CHECK: @align_wi = dso_local global i32 4
int align_wi = __alignof(wint_t);

// CHECK: @align_l = dso_local global i32 4
int align_l = __alignof(long);

// CHECK: @align_ll = dso_local global i32 8
int align_ll = __alignof(long long);

// CHECK: @align_p = dso_local global i32 4
int align_p = __alignof(void*);

// CHECK: @align_f = dso_local global i32 4
int align_f = __alignof(float);

// CHECK: @align_d = dso_local global i32 8
int align_d = __alignof(double);

// CHECK: @align_ld = dso_local global i32 16
int align_ld = __alignof(long double);

// CHECK: @align_vl = dso_local global i32 4
int align_vl = __alignof(va_list);

// CHECK: @align_a_c = dso_local global i32 1
int align_a_c = __alignof(_Atomic(char));

// CHECK: @align_a_s = dso_local global i32 2
int align_a_s = __alignof(_Atomic(short));

// CHECK: @align_a_i = dso_local global i32 4
int align_a_i = __alignof(_Atomic(int));

// CHECK: @align_a_wc = dso_local global i32 4
int align_a_wc = __alignof(_Atomic(wchar_t));

// CHECK: @align_a_wi = dso_local global i32 4
int align_a_wi = __alignof(_Atomic(wint_t));

// CHECK: @align_a_l = dso_local global i32 4
int align_a_l = __alignof(_Atomic(long));

// CHECK: @align_a_ll = dso_local global i32 8
int align_a_ll = __alignof(_Atomic(long long));

// CHECK: @align_a_p = dso_local global i32 4
int align_a_p = __alignof(_Atomic(void*));

// CHECK: @align_a_f = dso_local global i32 4
int align_a_f = __alignof(_Atomic(float));

// CHECK: @align_a_d = dso_local global i32 8
int align_a_d = __alignof(_Atomic(double));

// CHECK: @align_a_ld = dso_local global i32 16
int align_a_ld = __alignof(_Atomic(long double));

// CHECK: @align_a_s4 = dso_local global i32 4
int align_a_s4 = __alignof(_Atomic(struct { char s[4]; }));

// CHECK: @align_a_s8 = dso_local global i32 8
int align_a_s8 = __alignof(_Atomic(struct { char s[8]; }));

// CHECK: @align_a_s16 = dso_local global i32 16
int align_a_s16 = __alignof(_Atomic(struct { char s[16]; }));

// CHECK: @align_a_s32 = dso_local global i32 1
int align_a_s32 = __alignof(_Atomic(struct { char s[32]; }));


// Check Sizes

// CHECK: @size_a_c = dso_local global i32 1
int size_a_c = sizeof(_Atomic(char));

// CHECK: @size_a_s = dso_local global i32 2
int size_a_s = sizeof(_Atomic(short));

// CHECK: @size_a_i = dso_local global i32 4
int size_a_i = sizeof(_Atomic(int));

// CHECK: @size_a_wc = dso_local global i32 4
int size_a_wc = sizeof(_Atomic(wchar_t));

// CHECK: @size_a_wi = dso_local global i32 4
int size_a_wi = sizeof(_Atomic(wint_t));

// CHECK: @size_a_l = dso_local global i32 4
int size_a_l = sizeof(_Atomic(long));

// CHECK: @size_a_ll = dso_local global i32 8
int size_a_ll = sizeof(_Atomic(long long));

// CHECK: @size_a_p = dso_local global i32 4
int size_a_p = sizeof(_Atomic(void*));

// CHECK: @size_a_f = dso_local global i32 4
int size_a_f = sizeof(_Atomic(float));

// CHECK: @size_a_d = dso_local global i32 8
int size_a_d = sizeof(_Atomic(double));

// CHECK: @size_a_ld = dso_local global i32 16
int size_a_ld = sizeof(_Atomic(long double));


// Check types

// CHECK: zeroext i8 @check_char()
char check_char() { return 0; }

// CHECK: define dso_local signext i16 @check_short()
short check_short() { return 0; }

// CHECK: define dso_local i32 @check_int()
int check_int() { return 0; }

// CHECK: define dso_local i32 @check_wchar_t()
int check_wchar_t() { return 0; }

// CHECK: define dso_local i32 @check_long()
long check_long() { return 0; }

// CHECK: define dso_local i64 @check_longlong()
long long check_longlong() { return 0; }

// CHECK: define dso_local zeroext i8 @check_uchar()
unsigned char check_uchar() { return 0; }

// CHECK: define dso_local zeroext i16 @check_ushort()
unsigned short check_ushort() { return 0; }

// CHECK: define dso_local i32 @check_uint()
unsigned int check_uint() { return 0; }

// CHECK: define dso_local i32 @check_ulong()
unsigned long check_ulong() { return 0; }

// CHECK: define dso_local i64 @check_ulonglong()
unsigned long long check_ulonglong() { return 0; }

// CHECK: define dso_local i32 @check_size_t()
size_t check_size_t() { return 0; }

// CHECK: define dso_local float @check_float()
float check_float() { return 0; }

// CHECK: define dso_local double @check_double()
double check_double() { return 0; }

// CHECK: define dso_local fp128 @check_longdouble()
long double check_longdouble() { return 0; }
