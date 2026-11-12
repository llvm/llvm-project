// UNSUPPORTED: system-windows
// A basic clang -cc1 command-line, and simple environment check.

// RUN: %clang -### %s --target=riscv64 \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv64_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=CC1 %s
// CC1: "-cc1" "-triple" "riscv64"

// Test interaction with -fuse-ld=lld, if lld is available.
// RUN: %clang -### %s --target=riscv32 -fuse-ld=lld -B%S/Inputs/lld \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv64_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=LLD %s
// LLD: ld.lld

// In the below tests, --rtlib=platform is used so that the driver ignores
// the configure-time CLANG_DEFAULT_RTLIB option when choosing the runtime lib

// RUN: env "PATH=" %clang -### %s -fuse-ld= \
// RUN:   --target=riscv64-unknown-elf --rtlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv64_tree \
// RUN:   --sysroot=%S/Inputs/basic_riscv64_tree/riscv64-unknown-elf 2>&1 -no-pie \
// RUN:   | FileCheck -check-prefix=C-RV64-BAREMETAL-LP64 %s

// C-RV64-BAREMETAL-LP64: "{{.*}}Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../../../bin/riscv64-unknown-elf-ld"
// C-RV64-BAREMETAL-LP64: "--sysroot={{.*}}/Inputs/basic_riscv64_tree/riscv64-unknown-elf"
// C-RV64-BAREMETAL-LP64: "-X"
// C-RV64-BAREMETAL-LP64: "{{.*}}/Inputs/basic_riscv64_tree/riscv64-unknown-elf/lib/crt0.o"
// C-RV64-BAREMETAL-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/crtbegin.o"
// C-RV64-BAREMETAL-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1"
// C-RV64-BAREMETAL-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/riscv64-unknown-elf/lib"
// C-RV64-BAREMETAL-LP64: "--start-group" "-lgcc" "-lc" "-lgloss" "--end-group"
// C-RV64-BAREMETAL-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/crtend.o"

// RUN: env "PATH=" %clang -### %s -fuse-ld= \
// RUN:   --target=riscv64-unknown-elf --rtlib=platform \
// RUN:   --sysroot= \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv64_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64-BAREMETAL-NOSYSROOT-LP64 %s

// C-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../../../bin/riscv64-unknown-elf-ld"
// C-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../../../riscv64-unknown-elf/lib/crt0.o"
// C-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/crtbegin.o"
// C-RV64-BAREMETAL-NOSYSROOT-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1"
// C-RV64-BAREMETAL-NOSYSROOT-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../../../riscv64-unknown-elf/lib"
// C-RV64-BAREMETAL-NOSYSROOT-LP64: "--start-group" "-lgcc" "-lc" "-lgloss" "--end-group"
// C-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/crtend.o"

// RUN: env "PATH=" %clangxx -### %s -fuse-ld= \
// RUN:   --target=riscv64-unknown-elf -stdlib=libstdc++ --rtlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv64_tree \
// RUN:   --sysroot=%S/Inputs/basic_riscv64_tree/riscv64-unknown-elf 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-RV64-BAREMETAL-LP64 %s

// CXX-RV64-BAREMETAL-LP64: "-internal-isystem" "{{.*}}Inputs/basic_riscv64_tree/riscv64-unknown-elf/include/c++/8.0.1"
// CXX-RV64-BAREMETAL-LP64: "{{.*}}Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../../../bin/riscv64-unknown-elf-ld"
// CXX-RV64-BAREMETAL-LP64: "--sysroot={{.*}}/Inputs/basic_riscv64_tree/riscv64-unknown-elf"
// CXX-RV64-BAREMETAL-LP64: "{{.*}}/Inputs/basic_riscv64_tree/riscv64-unknown-elf/lib/crt0.o"
// CXX-RV64-BAREMETAL-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/crtbegin.o"
// CXX-RV64-BAREMETAL-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1"
// CXX-RV64-BAREMETAL-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/riscv64-unknown-elf/lib"
// CXX-RV64-BAREMETAL-LP64: "-lstdc++" "-lm" "--start-group" "-lgcc" "-lc" "-lgloss" "--end-group"
// CXX-RV64-BAREMETAL-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/crtend.o"

// RUN: env "PATH=" %clangxx -### %s -fuse-ld= \
// RUN:   --target=riscv64-unknown-elf -stdlib=libstdc++ --rtlib=platform \
// RUN:   --sysroot= \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv64_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-RV64-BAREMETAL-NOSYSROOT-LP64 %s

// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "-internal-isystem" "{{.*}}Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../../../riscv64-unknown-elf/include/c++/8.0.1"
// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../../../bin/riscv64-unknown-elf-ld"
// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../../../riscv64-unknown-elf/lib/crt0.o"
// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/crtbegin.o"
// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1"
// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "-L{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/../../../../riscv64-unknown-elf/lib"
// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "-lstdc++" "-lm" "--start-group" "-lgcc" "-lc" "-lgloss" "--end-group"
// CXX-RV64-BAREMETAL-NOSYSROOT-LP64: "{{.*}}/Inputs/basic_riscv64_tree/lib/gcc/riscv64-unknown-elf/8.0.1/crtend.o"

// RUN: env "PATH=" %clang -### %s -fuse-ld= -no-pie \
// RUN:   --target=riscv64-unknown-linux-gnu --rtlib=platform --unwindlib=platform -mabi=lp64 \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_linux_sdk \
// RUN:   --sysroot=%S/Inputs/multilib_riscv_linux_sdk/sysroot 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64-LINUX-MULTI-LP64 %s

// C-RV64-LINUX-MULTI-LP64: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/../../../../riscv64-unknown-linux-gnu/bin/ld"
// C-RV64-LINUX-MULTI-LP64: "--sysroot={{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot"
// C-RV64-LINUX-MULTI-LP64: "-m" "elf64lriscv" "-X"
// C-RV64-LINUX-MULTI-LP64: "-dynamic-linker" "/lib/ld-linux-riscv64-lp64.so.1"
// C-RV64-LINUX-MULTI-LP64: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib64/lp64/crtbegin.o"
// C-RV64-LINUX-MULTI-LP64: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib64/lp64"
// C-RV64-LINUX-MULTI-LP64: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/lib64/lp64"
// C-RV64-LINUX-MULTI-LP64: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/usr/lib64/lp64"

// RUN: env "PATH=" %clang -### %s -fuse-ld=ld -no-pie \
// RUN:   --target=riscv64-unknown-linux-gnu --rtlib=platform --unwindlib=platform -march=rv64imafd \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_linux_sdk \
// RUN:   --sysroot=%S/Inputs/multilib_riscv_linux_sdk/sysroot 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64-LINUX-MULTI-LP64D %s

// C-RV64-LINUX-MULTI-LP64D: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/../../../../riscv64-unknown-linux-gnu/bin/ld"
// C-RV64-LINUX-MULTI-LP64D: "--sysroot={{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot"
// C-RV64-LINUX-MULTI-LP64D: "-m" "elf64lriscv"
// C-RV64-LINUX-MULTI-LP64D: "-dynamic-linker" "/lib/ld-linux-riscv64-lp64d.so.1"
// C-RV64-LINUX-MULTI-LP64D: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib64/lp64d/crtbegin.o"
// C-RV64-LINUX-MULTI-LP64D: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/lib64/lp64d"
// C-RV64-LINUX-MULTI-LP64D: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/lib64/lp64d"
// C-RV64-LINUX-MULTI-LP64D: "-L{{.*}}/Inputs/multilib_riscv_linux_sdk/sysroot/usr/lib64/lp64d"

// RUN: env "PATH=" %clang -### %s -fuse-ld=ld \
// RUN:   --target=riscv64-unknown-elf --rtlib=platform --unwindlib=platform --sysroot= \
// RUN:   -march=rv64imac -mabi=lp64\
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64IMAC-BAREMETAL-MULTI-LP64 %s

// C-RV64IMAC-BAREMETAL-MULTI-LP64: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../../../riscv64-unknown-elf/bin/ld"
// C-RV64IMAC-BAREMETAL-MULTI-LP64: "-m" "elf64lriscv"
// C-RV64IMAC-BAREMETAL-MULTI-LP64: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../../../riscv64-unknown-elf/lib/rv64imac/lp64/crt0.o"
// C-RV64IMAC-BAREMETAL-MULTI-LP64: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/rv64imac/lp64/crtbegin.o"
// C-RV64IMAC-BAREMETAL-MULTI-LP64: "-L{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0"
// C-RV64IMAC-BAREMETAL-MULTI-LP64: "-L{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../../../riscv64-unknown-elf/lib"
// C-RV64IMAC-BAREMETAL-MULTI-LP64: "--start-group" "-lgcc" "-lc" "-lgloss" "--end-group"
// C-RV64IMAC-BAREMETAL-MULTI-LP64: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/rv64imac/lp64/crtend.o"

// RUN: env "PATH=" %clang -### %s -fuse-ld=ld \
// RUN:   --target=riscv64-unknown-elf --rtlib=platform --unwindlib=platform --sysroot= \
// RUN:   -march=rv64imafdc -mabi=lp64d \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64IMAFDC-BAREMETAL-MULTI-ILP64D %s

// C-RV64IMAFDC-BAREMETAL-MULTI-ILP64D: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../../../riscv64-unknown-elf/bin/ld"
// C-RV64IMAFDC-BAREMETAL-MULTI-ILP64D: "-m" "elf64lriscv"
// C-RV64IMAFDC-BAREMETAL-MULTI-ILP64D: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../../../riscv64-unknown-elf/lib/rv64imafdc/lp64d/crt0.o"
// C-RV64IMAFDC-BAREMETAL-MULTI-ILP64D: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/rv64imafdc/lp64d/crtbegin.o"
// C-RV64IMAFDC-BAREMETAL-MULTI-ILP64D: "-L{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0"
// C-RV64IMAFDC-BAREMETAL-MULTI-ILP64D: "-L{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/../../../../riscv64-unknown-elf/lib"
// C-RV64IMAFDC-BAREMETAL-MULTI-ILP64D: "--start-group" "-lgcc" "-lc" "-lgloss" "--end-group"
// C-RV64IMAFDC-BAREMETAL-MULTI-ILP64D: "{{.*}}/Inputs/multilib_riscv_elf_sdk/lib/gcc/riscv64-unknown-elf/8.2.0/rv64imafdc/lp64d/crtend.o"

// Check that --rtlib can be used to override the used runtime library
// RUN: %clang -### %s \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk \
// RUN:   --target=riscv64-unknown-elf --rtlib=libgcc --unwindlib=libgcc 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64-RTLIB-LIBGCC-LP64 %s
// C-RV64-RTLIB-LIBGCC-LP64: "{{.*}}crt0.o"
// C-RV64-RTLIB-LIBGCC-LP64: "{{.*}}crtbegin.o"
// C-RV64-RTLIB-LIBGCC-LP64: "--start-group" "-lgcc" "-lc" "-lgloss" "--end-group"
// C-RV64-RTLIB-LIBGCC-LP64: "{{.*}}crtend.o"

// RUN: %clang -### %s \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk \
// RUN:   --target=riscv64-unknown-elf --rtlib=compiler-rt --unwindlib=compiler-rt 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64-RTLIB-COMPILERRT-LP64 %s
// C-RV64-RTLIB-COMPILERRT-LP64: "{{.*}}crt0.o"
// C-RV64-RTLIB-COMPILERRT-LP64: "{{.*}}clang_rt.crtbegin.o"
// C-RV64-RTLIB-COMPILERRT-LP64: "--start-group" "{{.*}}libclang_rt.builtins.a" "-lc" "-lgloss" "--end-group"
// C-RV64-RTLIB-COMPILERRT-LP64: "{{.*}}clang_rt.crtend.o"

// RUN: %clang -### %s --target=riscv64 \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv64_tree --sysroot= \
// RUN:   -resource-dir=%s/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck -check-prefix=RESOURCE-INC %s
// RESOURCE-INC: "-internal-isystem" "{{.*}}/Inputs/resource_dir/include"
// RESOURCE-INC: "-internal-isystem" "{{.*}}/basic_riscv64_tree/{{.*}}riscv64-unknown-linux-gnu/include"

// RUN: %clang -### %s --target=riscv64 \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv64_tree --sysroot= \
// RUN:   -resource-dir=%s/Inputs/resource_dir -nobuiltininc 2>&1 \
// RUN:   | FileCheck -check-prefix=NO-RESOURCE-INC %s
// NO-RESOURCE-INC-NOT: "-internal-isystem" "{{.*}}Inputs/resource_dir/include"
// NO-RESOURCE-INC: "-internal-isystem" "{{.*}}/basic_riscv64_tree/{{.*}}riscv64-unknown-linux-gnu/include"

// RUN: %clang --target=riscv64 %s -emit-llvm -S -o - | FileCheck %s

/// Check that "--no-relax" is forwarded to the linker for RISC-V.
// RUN: env "PATH=" %clang %s -### 2>&1 -mno-relax \
// RUN:   --target=riscv64-unknown-elf --rtlib=platform --unwindlib=platform --sysroot= \
// RUN:   -march=rv64imac -mabi=lp64\
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-RV64-NORELAX %s
// CHECK-RV64-NORELAX: "--no-relax"

/// Check that "--no-relax" is not forwarded to the linker for RISC-V.
// RUN:env "PATH=" %clang %s -### 2>&1 \
// RUN:   --target=riscv64-unknown-elf --rtlib=platform --unwindlib=platform --sysroot= \
// RUN:   -march=rv64imac -mabi=lp64\
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_elf_sdk 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-RV64-RELAX %s
// CHECK-RV64-RELAX-NOT: "--no-relax"

/// Check that "--no-relax" is forwarded to the linker for RISC-V (Gnu.cpp).
// RUN: env "PATH=" %clang -### %s -fuse-ld=ld -no-pie -mno-relax \
// RUN:   --target=riscv64-unknown-linux-gnu --rtlib=platform --unwindlib=platform -mabi=lp64 \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_linux_sdk \
// RUN:   --sysroot=%S/Inputs/multilib_riscv_linux_sdk/sysroot 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-RV64-GNU-NORELAX %s
// CHECK-RV64-GNU-NORELAX: "--no-relax"

/// Check that "--no-relax" is not forwarded to the linker for RISC-V (Gnu.cpp).
// RUN: env "PATH=" %clang -### %s -fuse-ld=ld -no-pie \
// RUN:   --target=riscv64-unknown-linux-gnu --rtlib=platform --unwindlib=platform -mabi=lp64 \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_riscv_linux_sdk \
// RUN:   --sysroot=%S/Inputs/multilib_riscv_linux_sdk/sysroot 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-RV64-GNU-RELAX %s
// CHECK-RV64-GNU-RELAX-NOT: "--no-relax"

/// Check that "-static -pie" is forwarded to linker when "-static-pie" is used
// RUN: %clang -static-pie -### %s -fuse-ld= \
// RUN:   --target=riscv64-unknown-elf -rtlib=platform --unwindlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/basic_riscv64_tree \
// RUN:   --sysroot=%S/Inputs/basic_riscv64_tree/riscv64-unknown-elf 2>&1 \
// RUN:   | FileCheck -check-prefix=C-RV64-STATIC-PIE %s

// C-RV64-STATIC-PIE: "-Bstatic" "-pie" "--no-dynamic-linker" "-z" "text" "-m" "elf64lriscv" "-X"
// C-RV64-STATIC-PIE: "{{.*}}rcrt1.o"
// C-RV64-STATIC-PIE: "{{.*}}crtbeginS.o"
// C-RV64-STATIC-PIE: "--start-group" "-lgcc" "-lc" "-lgloss" "--end-group"
// C-RV64-STATIC-PIE: "{{.*}}crtendS.o"

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

// CHECK: @align_f16 = dso_local global i32 2
int align_f16 = __alignof(_Float16);

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

// CHECK: @align_a_f16 = dso_local global i32 2
int align_a_f16 = __alignof(_Atomic(_Float16));

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

// CHECK: @size_a_f16 = dso_local global i32 2
int size_a_f16 = sizeof(_Atomic(_Float16));

// CHECK: @size_a_f = dso_local global i32 4
int size_a_f = sizeof(_Atomic(float));

// CHECK: @size_a_d = dso_local global i32 8
int size_a_d = sizeof(_Atomic(double));

// CHECK: @size_a_ld = dso_local global i32 16
int size_a_ld = sizeof(_Atomic(long double));


// Check types

// CHECK: define dso_local zeroext i8 @check_char()
char check_char(void) { return 0; }

// CHECK: define dso_local signext i16 @check_short()
short check_short(void) { return 0; }

// CHECK: define dso_local signext i32 @check_int()
int check_int(void) { return 0; }

// CHECK: define dso_local signext i32 @check_wchar_t()
int check_wchar_t(void) { return 0; }

// CHECK: define dso_local i64 @check_long()
long check_long(void) { return 0; }

// CHECK: define dso_local i64 @check_longlong()
long long check_longlong(void) { return 0; }

// CHECK: define dso_local zeroext i8 @check_uchar()
unsigned char check_uchar(void) { return 0; }

// CHECK: define dso_local zeroext i16 @check_ushort()
unsigned short check_ushort(void) { return 0; }

// CHECK: define dso_local signext i32 @check_uint()
unsigned int check_uint(void) { return 0; }

// CHECK: define dso_local i64 @check_ulong()
unsigned long check_ulong(void) { return 0; }

// CHECK: define dso_local i64 @check_ulonglong()
unsigned long long check_ulonglong(void) { return 0; }

// CHECK: define dso_local i64 @check_size_t()
size_t check_size_t(void) { return 0; }

// CHECK: define dso_local half @check_float16()
_Float16 check_float16(void) { return 0; }

// CHECK: define dso_local float @check_float()
float check_float(void) { return 0; }

// CHECK: define dso_local double @check_double()
double check_double(void) { return 0; }

// CHECK: define dso_local fp128 @check_longdouble()
long double check_longdouble(void) { return 0; }
