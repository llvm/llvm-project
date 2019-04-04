// A basic clang -cc1 command-line, and simple environment check.

// RUN: %clang %s -### -no-canonical-prefixes -target riscv32 2>&1 | FileCheck -check-prefix=CC1 %s
// CC1: clang{{.*}} "-cc1" "-triple" "riscv32"

// RUN: %clang %s -### -no-canonical-prefixes \
// RUN:   -target riscv32-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv32_tree \
// RUN:   --sysroot=%S/Inputs/basic_riscv32_tree/riscv32-unknown-elf 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV32-BAREMETAL-ILP32 %s

// C-RV32-BAREMETAL-ILP32: "-fuse-init-array"
// C-RV32-BAREMETAL-ILP32: "{{.*}}Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1/../../../../bin{{/|\\\\}}riscv32-unknown-elf-ld"
// C-RV32-BAREMETAL-ILP32: "--sysroot={{.*}}/Inputs/basic_riscv32_tree/riscv32-unknown-elf"
// C-RV32-BAREMETAL-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/riscv32-unknown-elf/lib{{/|\\\\}}crt0.o"
// C-RV32-BAREMETAL-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1{{/|\\\\}}crtbegin.o"
// C-RV32-BAREMETAL-ILP32: "-L{{.*}}/Inputs/basic_riscv32_tree/riscv32-unknown-elf/lib"
// C-RV32-BAREMETAL-ILP32: "-L{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1"
// C-RV32-BAREMETAL-ILP32: "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// C-RV32-BAREMETAL-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1{{/|\\\\}}crtend.o"

// RUN: %clang %s -### -no-canonical-prefixes \
// RUN:   -target riscv32-unknown-elf \
// RUN:   --sysroot= \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv32_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV32-BAREMETAL-NOSYSROOT-ILP32 %s

// C-RV32-BAREMETAL-NOSYSROOT-ILP32: "-fuse-init-array"
// C-RV32-BAREMETAL-NOSYSROOT-ILP32: "{{.*}}Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1/../../../../bin{{/|\\\\}}riscv32-unknown-elf-ld"
// C-RV32-BAREMETAL-NOSYSROOT-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1/../../../../riscv32-unknown-elf/lib{{/|\\\\}}crt0.o"
// C-RV32-BAREMETAL-NOSYSROOT-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1{{/|\\\\}}crtbegin.o"
// C-RV32-BAREMETAL-NOSYSROOT-ILP32: "-L{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1/../../../../riscv32-unknown-elf{{/|\\\\}}lib"
// C-RV32-BAREMETAL-NOSYSROOT-ILP32: "-L{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1"
// C-RV32-BAREMETAL-NOSYSROOT-ILP32: "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// C-RV32-BAREMETAL-NOSYSROOT-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1{{/|\\\\}}crtend.o"

// RUN: %clangxx %s -### -no-canonical-prefixes \
// RUN:   -target riscv32-unknown-elf -stdlib=libstdc++ \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv32_tree \
// RUN:   --sysroot=%S/Inputs/basic_riscv32_tree/riscv32-unknown-elf 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-RV32-BAREMETAL-ILP32 %s

// CXX-RV32-BAREMETAL-ILP32: "-fuse-init-array"
// CXX-RV32-BAREMETAL-ILP32: "-internal-isystem" "{{.*}}Inputs/basic_riscv32_tree/riscv32-unknown-elf/include/c++{{/|\\\\}}8.0.1"
// CXX-RV32-BAREMETAL-ILP32: "{{.*}}Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1/../../../../bin{{/|\\\\}}riscv32-unknown-elf-ld"
// CXX-RV32-BAREMETAL-ILP32: "--sysroot={{.*}}/Inputs/basic_riscv32_tree/riscv32-unknown-elf"
// CXX-RV32-BAREMETAL-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/riscv32-unknown-elf/lib{{/|\\\\}}crt0.o"
// CXX-RV32-BAREMETAL-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1{{/|\\\\}}crtbegin.o"
// CXX-RV32-BAREMETAL-ILP32: "-L{{.*}}/Inputs/basic_riscv32_tree/riscv32-unknown-elf/lib"
// CXX-RV32-BAREMETAL-ILP32: "-L{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1"
// CXX-RV32-BAREMETAL-ILP32: "-lstdc++" "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// CXX-RV32-BAREMETAL-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1{{/|\\\\}}crtend.o"

// RUN: %clangxx %s -### -no-canonical-prefixes \
// RUN:   -target riscv32-unknown-elf -stdlib=libstdc++ \
// RUN:   --sysroot= \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv32_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-RV32-BAREMETAL-NOSYSROOT-ILP32 %s

// CXX-RV32-BAREMETAL-NOSYSROOT-ILP32: "-fuse-init-array"
// CXX-RV32-BAREMETAL-NOSYSROOT-ILP32: "-internal-isystem" "{{.*}}Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1/../../../../riscv32-unknown-elf/include/c++{{/|\\\\}}8.0.1"
// CXX-RV32-BAREMETAL-NOSYSROOT-ILP32: "{{.*}}Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1/../../../../bin{{/|\\\\}}riscv32-unknown-elf-ld"
// CXX-RV32-BAREMETAL-NOSYSROOT-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1/../../../../riscv32-unknown-elf/lib{{/|\\\\}}crt0.o"
// CXX-RV32-BAREMETAL-NOSYSROOT-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1{{/|\\\\}}crtbegin.o"
// CXX-RV32-BAREMETAL-NOSYSROOT-ILP32: "-L{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1/../../../../riscv32-unknown-elf/lib"
// CXX-RV32-BAREMETAL-NOSYSROOT-ILP32: "-L{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1"
// CXX-RV32-BAREMETAL-NOSYSROOT-ILP32: "-lstdc++" "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// CXX-RV32-BAREMETAL-NOSYSROOT-ILP32: "{{.*}}/Inputs/basic_riscv32_tree/lib/gcc/riscv32-unknown-elf/8.0.1{{/|\\\\}}crtend.o"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld=ld \
// RUN:   -target riscv32-unknown-linux-gnu \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_linux_sdk \
// RUN:   --sysroot=%S/Inputs/multilib_riscv_linux_sdk/sysroot 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV32-LINUX-MULTI-ILP32 %s

// C-RV32-LINUX-MULTI-ILP32: "-fuse-init-array"
// C-RV32-LINUX-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/../../../../riscv64-unknown-linux-gnu/bin{{/|\\\\}}ld"
// C-RV32-LINUX-MULTI-ILP32: "--sysroot={{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot"
// C-RV32-LINUX-MULTI-ILP32: "-m" "elf32lriscv"
// C-RV32-LINUX-MULTI-ILP32: "-dynamic-linker" "/lib/ld-linux-riscv32-ilp32.so.1"
// C-RV32-LINUX-MULTI-ILP32: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib32/ilp32{{/|\\\\}}crtbegin.o"
// C-RV32-LINUX-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib32/ilp32"
// C-RV32-LINUX-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/lib32/ilp32"
// C-RV32-LINUX-MULTI-ILP32: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/usr/lib32/ilp32"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld=ld \
// RUN:   -target riscv32-unknown-linux-gnu -march=rv32imafd -mabi=ilp32d \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_linux_sdk \
// RUN:   --sysroot=%S/Inputs/multilib_riscv_linux_sdk/sysroot 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV32-LINUX-MULTI-ILP32D %s

// C-RV32-LINUX-MULTI-ILP32D: "-fuse-init-array"
// C-RV32-LINUX-MULTI-ILP32D: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/../../../../riscv64-unknown-linux-gnu/bin{{/|\\\\}}ld"
// C-RV32-LINUX-MULTI-ILP32D: "--sysroot={{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot"
// C-RV32-LINUX-MULTI-ILP32D: "-m" "elf32lriscv"
// C-RV32-LINUX-MULTI-ILP32D: "-dynamic-linker" "/lib/ld-linux-riscv32-ilp32d.so.1"
// C-RV32-LINUX-MULTI-ILP32D: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib32/ilp32d{{/|\\\\}}crtbegin.o"
// C-RV32-LINUX-MULTI-ILP32D: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib32/ilp32d"
// C-RV32-LINUX-MULTI-ILP32D: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/lib32/ilp32d"
// C-RV32-LINUX-MULTI-ILP32D: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/usr/lib32/ilp32d"

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
