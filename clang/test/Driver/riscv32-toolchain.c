// A basic clang -cc1 command-line, and simple environment check.

// RUN: %clang %s -### -no-canonical-prefixes -target riscv32 2>&1 | FileCheck -check-prefix=CC1 %s
// CC1: clang{{.*}} "-cc1" "-triple" "riscv32"


// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld=ld \
// RUN:   -target riscv32-linux-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_linux_sdk \
// RUN:   --sysroot=%S/Inputs/multilib_riscv_linux_sdk/sysroot 2>&1 \
// RUN:   | FileCheck -check-prefix=CC1-RV32-LINUX-ILP32 %s

// CC1-RV32-LINUX-ILP32: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/../../../../riscv64-unknown-linux-gnu/bin{{/|\\\\}}ld"
// CC1-RV32-LINUX-ILP32: "--sysroot={{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot"
// CC1-RV32-LINUX-ILP32: "-m" "elf32lriscv"
// CC1-RV32-LINUX-ILP32: "-dynamic-linker" "/lib/ld-linux-riscv32-ilp32.so.1"
// CC1-RV32-LINUX-ILP32: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib32/ilp32{{/|\\\\}}crtbegin.o"
// CC1-RV32-LINUX-ILP32: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib32/ilp32"
// CC1-RV32-LINUX-ILP32: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/lib32/ilp32"
// CC1-RV32-LINUX-ILP32: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/usr/lib32/ilp32"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld=ld \
// RUN:   -target riscv32-linux-unknown-elf -march=rv32imafd -mabi=ilp32d \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_linux_sdk \
// RUN:   --sysroot=%S/Inputs/multilib_riscv_linux_sdk/sysroot 2>&1 \
// RUN:   | FileCheck -check-prefix=CC1-RV32-LINUX-ILP32D %s

// CC1-RV32-LINUX-ILP32D: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/../../../../riscv64-unknown-linux-gnu/bin{{/|\\\\}}ld"
// CC1-RV32-LINUX-ILP32D: "--sysroot={{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot"
// CC1-RV32-LINUX-ILP32D: "-m" "elf32lriscv"
// CC1-RV32-LINUX-ILP32D: "-dynamic-linker" "/lib/ld-linux-riscv32-ilp32d.so.1"
// CC1-RV32-LINUX-ILP32D: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib32/ilp32d{{/|\\\\}}crtbegin.o"
// CC1-RV32-LINUX-ILP32D: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib32/ilp32d"
// CC1-RV32-LINUX-ILP32D: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/lib32/ilp32d"
// CC1-RV32-LINUX-ILP32D: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/usr/lib32/ilp32d"

// RUN: %clang -target riscv32 %s -emit-llvm -S -o - | FileCheck %s

typedef __builtin_va_list va_list;
typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef __WCHAR_TYPE__ wchar_t;

// CHECK: @align_c = global i32 1
int align_c = __alignof(char);

// CHECK: @align_s = global i32 2
int align_s = __alignof(short);

// CHECK: @align_i = global i32 4
int align_i = __alignof(int);

// CHECK: @align_wc = global i32 4
int align_wc = __alignof(wchar_t);

// CHECK: @align_l = global i32 4
int align_l = __alignof(long);

// CHECK: @align_ll = global i32 8
int align_ll = __alignof(long long);

// CHECK: @align_p = global i32 4
int align_p = __alignof(void*);

// CHECK: @align_f = global i32 4
int align_f = __alignof(float);

// CHECK: @align_d = global i32 8
int align_d = __alignof(double);

// CHECK: @align_ld = global i32 16
int align_ld = __alignof(long double);

// CHECK: @align_vl = global i32 4
int align_vl = __alignof(va_list);
