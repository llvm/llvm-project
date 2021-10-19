// A basic clang -cc1 command-line, and simple environment check.

// RUN: %clang %s -### -no-canonical-prefixes -target riscv64 \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv64_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=CC1 %s
// CC1: clang{{.*}} "-cc1" "-triple" "riscv64"

// Test interaction with -fuse-ld=lld, if lld is available.
// RUN: %clang %s -### -no-canonical-prefixes -target riscv32 -fuse-ld=lld \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv64_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=LLD %s
// LLD: {{(error: invalid linker name in argument '-fuse-ld=lld')|(ld.lld)}}

// In the below tests, --rtlib=platform is used so that the driver ignores
// the configure-time CLANG_DEFAULT_RTLIB option when choosing the runtime lib

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld= \
// RUN:   -target riscv64-unknown-elf --rtlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv64_tree \
// RUN:   --sysroot=%S/Inputs/basic_riscv64_tree/riscv64-unknown-elf 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64-BAREMETAL-LP64 %s

// C-RV64-BAREMETAL-LP64: "{{.*}}Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv64-unknown-elf-ld"
// C-RV64-BAREMETAL-LP64: "--sysroot={{.*}}/Inputs/basic_riscv64_tree/riscv64-unknown-elf"
// C-RV64-BAREMETAL-LP64: "{{.*}}/Inputs/basic_riscv64_tree/riscv64-unknown-elf/lib{{/|\\\\}}crt0.o"
// C-RV64-BAREMETAL-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1{{/|\\\\}}crtbegin.o"
// C-RV64-BAREMETAL-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1"
// C-RV64-BAREMETAL-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/riscv64-unknown-elf/lib"
// C-RV64-BAREMETAL-LP64: "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// C-RV64-BAREMETAL-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1{{/|\\\\}}crtend.o"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld= \
// RUN:   -target riscv64-unknown-elf --rtlib=platform \
// RUN:   --sysroot= \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv64_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64-BAREMETAL-NOSYSROOT-LP64 %s

// C-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv64-unknown-elf-ld"
// C-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/lib{{/|\\\\}}crt0.o"
// C-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1{{/|\\\\}}crtbegin.o"
// C-RV64-BAREMETAL-NOSYSROOT-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1"
// C-RV64-BAREMETAL-NOSYSROOT-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf{{/|\\\\}}lib"
// C-RV64-BAREMETAL-NOSYSROOT-LP64: "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// C-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1{{/|\\\\}}crtend.o"

// RUN: %clangxx %s -### -no-canonical-prefixes -fuse-ld= \
// RUN:   -target riscv64-unknown-elf -stdlib=libstdc++ --rtlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv64_tree \
// RUN:   --sysroot=%S/Inputs/basic_riscv64_tree/riscv64-unknown-elf 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-RV64-BAREMETAL-LP64 %s

// CXX-RV64-BAREMETAL-LP64: "-internal-isystem" "{{.*}}Inputs/basic_riscv64_tree/riscv64-unknown-elf/include/c++{{/|\\\\}}8.0.1"
// CXX-RV64-BAREMETAL-LP64: "{{.*}}Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv64-unknown-elf-ld"
// CXX-RV64-BAREMETAL-LP64: "--sysroot={{.*}}/Inputs/basic_riscv64_tree/riscv64-unknown-elf"
// CXX-RV64-BAREMETAL-LP64: "{{.*}}/Inputs/basic_riscv64_tree/riscv64-unknown-elf/lib{{/|\\\\}}crt0.o"
// CXX-RV64-BAREMETAL-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1{{/|\\\\}}crtbegin.o"
// CXX-RV64-BAREMETAL-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1"
// CXX-RV64-BAREMETAL-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/riscv64-unknown-elf/lib"
// CXX-RV64-BAREMETAL-LP64: "-lstdc++" "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// CXX-RV64-BAREMETAL-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1{{/|\\\\}}crtend.o"

// RUN: %clangxx %s -### -no-canonical-prefixes -fuse-ld= \
// RUN:   -target riscv64-unknown-elf -stdlib=libstdc++ --rtlib=platform \
// RUN:   --sysroot= \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv64_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-RV64-BAREMETAL-NOSYSROOT-LP64 %s

// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "-internal-isystem" "{{.*}}Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/include/c++{{/|\\\\}}8.0.1"
// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../..{{/|\\\\}}..{{/|\\\\}}bin{{/|\\\\}}riscv64-unknown-elf-ld"
// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/lib{{/|\\\\}}crt0.o"
// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1{{/|\\\\}}crtbegin.o"
// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1"
// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/lib"
// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "-lstdc++" "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1{{/|\\\\}}crtend.o"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld=ld -fuse-ld= \
// RUN:   -target riscv64-unknown-linux-gnu --rtlib=platform -mabi=lp64 \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_linux_sdk \
// RUN:   --sysroot=%S/Inputs/multilib_riscv_linux_sdk/sysroot 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64-LINUX-MULTI-LP64 %s

// C-RV64-LINUX-MULTI-LP64: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-linux-gnu/bin{{/|\\\\}}ld"
// C-RV64-LINUX-MULTI-LP64: "--sysroot={{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot"
// C-RV64-LINUX-MULTI-LP64: "-m" "elf64lriscv"
// C-RV64-LINUX-MULTI-LP64: "-dynamic-linker" "/lib/ld-linux-riscv64-lp64.so.1"
// C-RV64-LINUX-MULTI-LP64: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib64/lp64{{/|\\\\}}crtbegin.o"
// C-RV64-LINUX-MULTI-LP64: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib64/lp64"
// C-RV64-LINUX-MULTI-LP64: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/lib64/lp64"
// C-RV64-LINUX-MULTI-LP64: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/usr/lib64/lp64"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld=ld \
// RUN:   -target riscv64-unknown-linux-gnu --rtlib=platform -march=rv64imafd \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_linux_sdk \
// RUN:   --sysroot=%S/Inputs/multilib_riscv_linux_sdk/sysroot 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64-LINUX-MULTI-LP64D %s

// C-RV64-LINUX-MULTI-LP64D: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-linux-gnu/bin{{/|\\\\}}ld"
// C-RV64-LINUX-MULTI-LP64D: "--sysroot={{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot"
// C-RV64-LINUX-MULTI-LP64D: "-m" "elf64lriscv"
// C-RV64-LINUX-MULTI-LP64D: "-dynamic-linker" "/lib/ld-linux-riscv64-lp64d.so.1"
// C-RV64-LINUX-MULTI-LP64D: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib64/lp64d{{/|\\\\}}crtbegin.o"
// C-RV64-LINUX-MULTI-LP64D: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib64/lp64d"
// C-RV64-LINUX-MULTI-LP64D: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/lib64/lp64d"
// C-RV64-LINUX-MULTI-LP64D: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/usr/lib64/lp64d"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld=ld \
// RUN:   -target riscv64-unknown-elf --rtlib=platform --sysroot= \
// RUN:   -march=rv64imac -mabi=lp64\
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64IMAC-BAREMETAL-MULTI-LP64 %s

// C-RV64IMAC-BAREMETAL-MULTI-LP64: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/bin{{/|\\\\}}ld"
// C-RV64IMAC-BAREMETAL-MULTI-LP64: "-m" "elf64lriscv"
// C-RV64IMAC-BAREMETAL-MULTI-LP64: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/lib/rv64imac/lp64{{/|\\\\}}crt0.o"
// C-RV64IMAC-BAREMETAL-MULTI-LP64: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/rv64imac/lp64{{/|\\\\}}crtbegin.o"
// C-RV64IMAC-BAREMETAL-MULTI-LP64: "-L{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0"
// C-RV64IMAC-BAREMETAL-MULTI-LP64: "-L{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/lib"
// C-RV64IMAC-BAREMETAL-MULTI-LP64: "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// C-RV64IMAC-BAREMETAL-MULTI-LP64: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/rv64imac/lp64{{/|\\\\}}crtend.o"

// RUN: %clang %s -### -no-canonical-prefixes -fuse-ld=ld \
// RUN:   -target riscv64-unknown-elf --rtlib=platform --sysroot= \
// RUN:   -march=rv64imafdc -mabi=lp64d \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64IMAFDC-BAREMETAL-MULTI-ILP64D %s

// C-RV64IMAFDC-BAREMETAL-MULTI-ILP64D: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/bin{{/|\\\\}}ld"
// C-RV64IMAFDC-BAREMETAL-MULTI-ILP64D: "-m" "elf64lriscv"
// C-RV64IMAFDC-BAREMETAL-MULTI-ILP64D: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/lib/rv64imafdc/lp64d{{/|\\\\}}crt0.o"
// C-RV64IMAFDC-BAREMETAL-MULTI-ILP64D: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/rv64imafdc/lp64d{{/|\\\\}}crtbegin.o"
// C-RV64IMAFDC-BAREMETAL-MULTI-ILP64D: "-L{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0"
// C-RV64IMAFDC-BAREMETAL-MULTI-ILP64D: "-L{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../..{{/|\\\\}}..{{/|\\\\}}riscv64-unknown-elf/lib"
// C-RV64IMAFDC-BAREMETAL-MULTI-ILP64D: "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// C-RV64IMAFDC-BAREMETAL-MULTI-ILP64D: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/rv64imafdc/lp64d{{/|\\\\}}crtend.o"

// Check that --rtlib can be used to override the used runtime library
// RUN: %clang %s -### -no-canonical-prefixes \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk \
// RUN:   -target riscv64-unknown-elf --rtlib=libgcc 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64-RTLIB-LIBGCC-LP64 %s
// C-RV64-RTLIB-LIBGCC-LP64: "{{.*}}crt0.o"
// C-RV64-RTLIB-LIBGCC-LP64: "{{.*}}crtbegin.o"
// C-RV64-RTLIB-LIBGCC-LP64: "--start-group" "-lc" "-lgloss" "--end-group" "-lgcc"
// C-RV64-RTLIB-LIBGCC-LP64: "{{.*}}crtend.o"

// RUN: %clang %s -### -no-canonical-prefixes \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk \
// RUN:   -target riscv64-unknown-elf --rtlib=compiler-rt 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64-RTLIB-COMPILERRT-LP64 %s
// C-RV64-RTLIB-COMPILERRT-LP64: "{{.*}}crt0.o"
// C-RV64-RTLIB-COMPILERRT-LP64: "{{.*}}clang_rt.crtbegin-riscv64.o"
// C-RV64-RTLIB-COMPILERRT-LP64: "--start-group" "-lc" "-lgloss" "--end-group" "{{.*}}libclang_rt.builtins-riscv64.a"
// C-RV64-RTLIB-COMPILERRT-LP64: "{{.*}}clang_rt.crtend-riscv64.o"

// RUN: %clang -target riscv64 %s -emit-llvm -S -o - | FileCheck %s

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

// CHECK: @align_a_l = dso_local global i32 8
int align_a_l = __alignof(_Atomic(long));

// CHECK: @align_a_ll = dso_local global i32 8
int align_a_ll = __alignof(_Atomic(long long));

// CHECK: @align_a_p = dso_local global i32 8
int align_a_p = __alignof(_Atomic(void*));

// CHECK: @align_a_f = dso_local global i32 4
int align_a_f = __alignof(_Atomic(float));

// CHECK: @align_a_d = dso_local global i32 8
int align_a_d = __alignof(_Atomic(double));

// CHECK: @align_a_ld = dso_local global i32 16
int align_a_ld = __alignof(_Atomic(long double));

// CHECK: @align_a_s4 = dso_local global i32 4
int align_a_s4 = __alignof(_Atomic(struct { char _[4]; }));

// CHECK: @align_a_s8 = dso_local global i32 8
int align_a_s8 = __alignof(_Atomic(struct { char _[8]; }));

// CHECK: @align_a_s16 = dso_local global i32 16
int align_a_s16 = __alignof(_Atomic(struct { char _[16]; }));

// CHECK: @align_a_s32 = dso_local global i32 1
int align_a_s32 = __alignof(_Atomic(struct { char _[32]; }));


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

// CHECK: @size_a_l = dso_local global i32 8
int size_a_l = sizeof(_Atomic(long));

// CHECK: @size_a_ll = dso_local global i32 8
int size_a_ll = sizeof(_Atomic(long long));

// CHECK: @size_a_p = dso_local global i32 8
int size_a_p = sizeof(_Atomic(void*));

// CHECK: @size_a_f = dso_local global i32 4
int size_a_f = sizeof(_Atomic(float));

// CHECK: @size_a_d = dso_local global i32 8
int size_a_d = sizeof(_Atomic(double));

// CHECK: @size_a_ld = dso_local global i32 16
int size_a_ld = sizeof(_Atomic(long double));


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
