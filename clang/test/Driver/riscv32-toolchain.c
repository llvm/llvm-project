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

// CHECK: @align_c = dso_local global i32 1
int align_c = __alignof(char);

// CHECK: @align_s = dso_local global i32 2
int align_s = __alignof(short);

// CHECK: @align_i = dso_local global i32 4
int align_i = __alignof(int);

// CHECK: @align_wc = dso_local global i32 4
int align_wc = __alignof(wchar_t);

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
