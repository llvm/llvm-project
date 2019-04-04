// A basic clang -cc1 command-line, and simple environment check.

// RUN: %clang %s -### -no-canonical-prefixes -target riscv64 2>&1 | FileCheck -check-prefix=CC1 %s
// CC1: clang{{.*}} "-cc1" "-triple" "riscv64"

// RUN: %clang %s -### -no-canonical-prefixes \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv64_tree \
// RUN:   --sysroot=%S/Inputs/basic_riscv64_tree/riscv64-unknown-elf 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64-BAREMETAL-LP64 %s

// C-RV64-BAREMETAL-LP64: "-fuse-init-array"
// C-RV64-BAREMETAL-LP64: "{{.*}}Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../../../bin{{/|\\\\}}riscv64-unknown-elf-ld"
// C-RV64-BAREMETAL-LP64: "--sysroot={{.*}}/Inputs/basic_riscv64_tree/riscv64-unknown-elf"
// C-RV64-BAREMETAL-LP64: "{{.*}}/Inputs/basic_riscv64_tree/riscv64-unknown-elf/lib{{/|\\\\}}crt0.o"
// C-RV64-BAREMETAL-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1{{/|\\\\}}crtbegin.o"
// C-RV64-BAREMETAL-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/riscv64-unknown-elf/lib"
// C-RV64-BAREMETAL-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1"
// C-RV64-BAREMETAL-LP64: "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// C-RV64-BAREMETAL-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1{{/|\\\\}}crtend.o"

// RUN: %clang %s -### -no-canonical-prefixes \
// RUN:   -target riscv64-unknown-elf \
// RUN:   --sysroot= \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv64_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64-BAREMETAL-NOSYSROOT-LP64 %s

// C-RV64-BAREMETAL-NOSYSROOT-LP64: "-fuse-init-array"
// C-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../../../bin{{/|\\\\}}riscv64-unknown-elf-ld"
// C-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../../../riscv64-unknown-elf/lib{{/|\\\\}}crt0.o"
// C-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1{{/|\\\\}}crtbegin.o"
// C-RV64-BAREMETAL-NOSYSROOT-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../../../riscv64-unknown-elf{{/|\\\\}}lib"
// C-RV64-BAREMETAL-NOSYSROOT-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1"
// C-RV64-BAREMETAL-NOSYSROOT-LP64: "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// C-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1{{/|\\\\}}crtend.o"

// RUN: %clangxx %s -### -no-canonical-prefixes \
// RUN:   -target riscv64-unknown-elf -stdlib=libstdc++ \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv64_tree \
// RUN:   --sysroot=%S/Inputs/basic_riscv64_tree/riscv64-unknown-elf 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-RV64-BAREMETAL-LP64 %s

// CXX-RV64-BAREMETAL-LP64: "-fuse-init-array"
// CXX-RV64-BAREMETAL-LP64: "-internal-isystem" "{{.*}}Inputs/basic_riscv64_tree/riscv64-unknown-elf/include/c++{{/|\\\\}}8.0.1"
// CXX-RV64-BAREMETAL-LP64: "{{.*}}Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../../../bin{{/|\\\\}}riscv64-unknown-elf-ld"
// CXX-RV64-BAREMETAL-LP64: "--sysroot={{.*}}/Inputs/basic_riscv64_tree/riscv64-unknown-elf"
// CXX-RV64-BAREMETAL-LP64: "{{.*}}/Inputs/basic_riscv64_tree/riscv64-unknown-elf/lib{{/|\\\\}}crt0.o"
// CXX-RV64-BAREMETAL-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1{{/|\\\\}}crtbegin.o"
// CXX-RV64-BAREMETAL-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/riscv64-unknown-elf/lib"
// CXX-RV64-BAREMETAL-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1"
// CXX-RV64-BAREMETAL-LP64: "-lstdc++" "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// CXX-RV64-BAREMETAL-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1{{/|\\\\}}crtend.o"

// RUN: %clangxx %s -### -no-canonical-prefixes \
// RUN:   -target riscv64-unknown-elf -stdlib=libstdc++ \
// RUN:   --sysroot= \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv64_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-RV64-BAREMETAL-NOSYSROOT-LP64 %s

// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "-fuse-init-array"
// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "-internal-isystem" "{{.*}}Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../../../riscv64-unknown-elf/include/c++{{/|\\\\}}8.0.1"
// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../../../bin{{/|\\\\}}riscv64-unknown-elf-ld"
// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../../../riscv64-unknown-elf/lib{{/|\\\\}}crt0.o"
// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1{{/|\\\\}}crtbegin.o"
// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../../../riscv64-unknown-elf/lib"
// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1"
// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "-lstdc++" "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1{{/|\\\\}}crtend.o"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld=ld \
// RUN:   -target riscv64-unknown-linux-gnu \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_linux_sdk \
// RUN:   --sysroot=%S/Inputs/multilib_riscv_linux_sdk/sysroot 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64-LINUX-MULTI-LP64 %s

// C-RV64-LINUX-MULTI-LP64: "-fuse-init-array"
// C-RV64-LINUX-MULTI-LP64: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/../../../../riscv64-unknown-linux-gnu/bin{{/|\\\\}}ld"
// C-RV64-LINUX-MULTI-LP64: "--sysroot={{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot"
// C-RV64-LINUX-MULTI-LP64: "-m" "elf64lriscv"
// C-RV64-LINUX-MULTI-LP64: "-dynamic-linker" "/lib/ld-linux-riscv64-lp64.so.1"
// C-RV64-LINUX-MULTI-LP64: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib64/lp64{{/|\\\\}}crtbegin.o"
// C-RV64-LINUX-MULTI-LP64: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib64/lp64"
// C-RV64-LINUX-MULTI-LP64: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/lib64/lp64"
// C-RV64-LINUX-MULTI-LP64: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/usr/lib64/lp64"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld=ld \
// RUN:   -target riscv64-unknown-linux-gnu -march=rv64imafd -mabi=lp64d \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_linux_sdk \
// RUN:   --sysroot=%S/Inputs/multilib_riscv_linux_sdk/sysroot 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64-LINUX-MULTI-LP64D %s

// C-RV64-LINUX-MULTI-LP64D: "-fuse-init-array"
// C-RV64-LINUX-MULTI-LP64D: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/../../../../riscv64-unknown-linux-gnu/bin{{/|\\\\}}ld"
// C-RV64-LINUX-MULTI-LP64D: "--sysroot={{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot"
// C-RV64-LINUX-MULTI-LP64D: "-m" "elf64lriscv"
// C-RV64-LINUX-MULTI-LP64D: "-dynamic-linker" "/lib/ld-linux-riscv64-lp64d.so.1"
// C-RV64-LINUX-MULTI-LP64D: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib64/lp64d{{/|\\\\}}crtbegin.o"
// C-RV64-LINUX-MULTI-LP64D: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib64/lp64d"
// C-RV64-LINUX-MULTI-LP64D: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/lib64/lp64d"
// C-RV64-LINUX-MULTI-LP64D: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/usr/lib64/lp64d"

// RUN: %clang -target riscv64 %s -emit-llvm -S -o - | FileCheck %s

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

// CHECK: @align_l = dso_local global i32 8
int align_l = __alignof(long);

// CHECK: @align_ll = dso_local global i32 8
int align_ll = __alignof(long long);

// CHECK: @align_p = dso_local global i32 8
int align_p = __alignof(void*);

// CHECK: @align_f = dso_local global i32 4
int align_f = __alignof(float);

// CHECK: @align_d = dso_local global i32 8
int align_d = __alignof(double);

// CHECK: @align_ld = dso_local global i32 16
int align_ld = __alignof(long double);

// CHECK: @align_vl = dso_local global i32 8
int align_vl = __alignof(va_list);

// Check types

// CHECK: define dso_local zeroext i8 @check_char()
char check_char() { return 0; }

// CHECK: define dso_local signext i16 @check_short()
short check_short() { return 0; }

// CHECK: define dso_local signext i32 @check_int()
int check_int() { return 0; }

// CHECK: define dso_local signext i32 @check_wchar_t()
int check_wchar_t() { return 0; }

// CHECK: define dso_local i64 @check_long()
long check_long() { return 0; }

// CHECK: define dso_local i64 @check_longlong()
long long check_longlong() { return 0; }

// CHECK: define dso_local zeroext i8 @check_uchar()
unsigned char check_uchar() { return 0; }

// CHECK: define dso_local zeroext i16 @check_ushort()
unsigned short check_ushort() { return 0; }

// CHECK: define dso_local signext i32 @check_uint()
unsigned int check_uint() { return 0; }

// CHECK: define dso_local i64 @check_ulong()
unsigned long check_ulong() { return 0; }

// CHECK: define dso_local i64 @check_ulonglong()
unsigned long long check_ulonglong() { return 0; }

// CHECK: define dso_local i64 @check_size_t()
size_t check_size_t() { return 0; }

// CHECK: define dso_local float @check_float()
float check_float() { return 0; }

// CHECK: define dso_local double @check_double()
double check_double() { return 0; }

// CHECK: define dso_local fp128 @check_longdouble()
long double check_longdouble() { return 0; }
