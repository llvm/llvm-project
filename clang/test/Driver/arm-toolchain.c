// UNSUPPORTED: system-windows

// Test interaction with -fuse-ld=lld
// RUN: %clang -### %s -fuse-ld=lld \
// RUN:   --target=armv6m-none-eabi  \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree 2>&1 \
// RUN:   | FileCheck -check-prefix=LLD %s
// LLD: ld.lld

// RUN: %clang -### %s -fuse-ld= \
// RUN:   --target=armv6m-none-eabi --rtlib=libgcc \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree \
// RUN:   --sysroot=%S/Inputs/basic_arm_gcc_tree/armv6m-none-eabi 2>&1 \
// RUN:   | FileCheck -check-prefix=C-ARM-BAREMETAL %s

// C-ARM-BAREMETAL: "-cc1" "-triple" "thumbv6m-unknown-none-eabi"
// C-ARM-BAREMETAL: "-isysroot" "{{.*}}Inputs/basic_arm_gcc_tree/armv6m-none-eabi"
// C-ARM-BAREMETAL: "-internal-isystem" "{{.*}}Inputs/basic_arm_gcc_tree/armv6m-none-eabi/include"
// C-ARM-BAREMETAL: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../bin/armv6m-none-eabi-ld"
// C-ARM-BAREMETAL: "-Bstatic" "-EL"
// C-ARM-BAREMETAL: "{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/lib/crt0.o"
// C-ARM-BAREMETAL: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtbegin.o"
// C-ARM-BAREMETAL: "-L{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1"
// C-ARM-BAREMETAL: "-L{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/lib"
// C-ARM-BAREMETAL: "{{.*}}.o" "--start-group" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "-lc" "-lgloss" "--end-group"
// C-ARM-BAREMETAL: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtend.o"

// RUN: %clang -### %s -fuse-ld= \
// RUN:   --target=armv6m-none-eabi --rtlib=libgcc \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree \
// RUN:   --sysroot=  2>&1 \
// RUN:   | FileCheck -check-prefix=C-ARM-BAREMETAL-NOSYSROOT %s

// C-ARM-BAREMETAL-NOSYSROOT: "-cc1" "-triple" "thumbv6m-unknown-none-eabi"
// C-ARM-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/include"
// C-ARM-BAREMETAL-NOSYSROOT: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../bin/armv6m-none-eabi-ld"
// C-ARM-BAREMETAL-NOSYSROOT: "-Bstatic" "-EL"
// C-ARM-BAREMETAL-NOSYSROOT: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/lib/crt0.o"
// C-ARM-BAREMETAL-NOSYSROOT: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtbegin.o"
// C-ARM-BAREMETAL-NOSYSROOT: "-L{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1"
// C-ARM-BAREMETAL-NOSYSROOT: "-L{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/lib"
// C-ARM-BAREMETAL-NOSYSROOT: "{{.*}}.o" "--start-group" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "-lc" "-lgloss" "--end-group"
// C-ARM-BAREMETAL-NOSYSROOT: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtend.o"

// RUN: %clangxx -### %s -fuse-ld= \
// RUN:   --target=armv6m-none-eabi -stdlib=libstdc++ --rtlib=libgcc \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree \
// RUN:   --sysroot=%S/Inputs/basic_arm_gcc_tree/armv6m-none-eabi 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-ARM-BAREMETAL %s

// CXX-ARM-BAREMETAL: "-isysroot" "{{.*}}Inputs/basic_arm_gcc_tree/armv6m-none-eabi"
// CXX-ARM-BAREMETAL: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/include/c++/8.2.1/armv6m-none-eabi"
// CXX-ARM-BAREMETAL: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/include/c++/8.2.1/backward"
// CXX-ARM-BAREMETAL: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/include/c++/8.2.1"
// CXX-ARM-BAREMETAL: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/include"
// CXX-ARM-BAREMETAL: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../bin/armv6m-none-eabi-ld"
// CXX-ARM-BAREMETAL: "-Bstatic" "-EL"
// CXX-ARM-BAREMETAL: "{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/lib/crt0.o"
// CXX-ARM-BAREMETAL: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtbegin.o"
// CXX-ARM-BAREMETAL: "-L{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1"
// CXX-ARM-BAREMETAL: "-L{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/lib"
// CXX-ARM-BAREMETAL: "{{.*}}.o" "-lstdc++" "-lm" "--start-group" "-lgcc_s" "-lgcc" "-lc" "-lgloss" "--end-group"
// CXX-ARM-BAREMETAL: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtend.o"


// RUN: %clangxx -### %s -fuse-ld= \
// RUN:   --target=armv6m-none-eabi -stdlib=libstdc++ --rtlib=libgcc \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree \
// RUN:   --sysroot=  2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-ARM-BAREMETAL-NOSYSROOT %s

// CXX-ARM-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/include/c++/8.2.1/armv6m-none-eabi"
// CXX-ARM-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/include/c++/8.2.1/backward"
// CXX-ARM-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/include/c++/8.2.1"
// CXX-ARM-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/include"
// CXX-ARM-BAREMETAL-NOSYSROOT: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../bin/armv6m-none-eabi-ld"
// CXX-ARM-BAREMETAL-NOSYSROOT: "-Bstatic" "-EL"
// CXX-ARM-BAREMETAL-NOSYSROOT: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/lib/crt0.o"
// CXX-ARM-BAREMETAL-NOSYSROOT: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtbegin.o"
// CXX-ARM-BAREMETAL-NOSYSROOT: "-L{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1"
// CXX-ARM-BAREMETAL-NOSYSROOT: "-L{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/lib"
// CXX-ARM-BAREMETAL-NOSYSROOT: "{{.*}}.o" "-lstdc++" "-lm" "--start-group" "-lgcc_s" "-lgcc" "-lc" "-lgloss" "--end-group"
// CXX-ARM-BAREMETAL-NOSYSROOT: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtend.o"

// RUN: %clangxx -### %s -fuse-ld= \
// RUN:   --target=armv6m-none-eabi -stdlib=libc++ --rtlib=libgcc \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree \
// RUN:   --sysroot=%S/Inputs/basic_arm_gcc_tree/armv6m-none-eabi 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-ARM-BAREMETAL-LIBCXX %s

// CXX-ARM-BAREMETAL-LIBCXX: "-isysroot" "{{.*}}Inputs/basic_arm_gcc_tree/armv6m-none-eabi"
// CXX-ARM-BAREMETAL-LIBCXX: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/include/c++/v1"
// CXX-ARM-BAREMETAL-LIBCXX: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/include"
// CXX-ARM-BAREMETAL-LIBCXX: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../bin/armv6m-none-eabi-ld"
// CXX-ARM-BAREMETAL-LIBCXX: "-Bstatic" "-EL"
// CXX-ARM-BAREMETAL-LIBCXX: "{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/lib/crt0.o"
// CXX-ARM-BAREMETAL-LIBCXX: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtbegin.o"
// CXX-ARM-BAREMETAL-LIBCXX: "-L{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1"
// CXX-ARM-BAREMETAL-LIBCXX: "-L{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/lib"
// CXX-ARM-BAREMETAL-LIBCXX: "{{.*}}.o" "-lc++" "-lm" "--start-group" "-lgcc_s" "-lgcc" "-lc" "-lgloss" "--end-group"
// CXX-ARM-BAREMETAL-LIBCXX: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtend.o"

// RUN: %clangxx -### %s -fuse-ld= \
// RUN:   --target=armv6m-none-eabi -stdlib=libc++ --rtlib=libgcc \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree \
// RUN:   --sysroot=  2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX %s

// CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/include/c++/v1"
// CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/include"
// CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../bin/armv6m-none-eabi-ld"
// CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX: "-Bstatic" "-EL"
// CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/lib/crt0.o"
// CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtbegin.o"
// CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX: "-L{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1"
// CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX: "-L{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/lib"
// CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX: "{{.*}}.o" "-lc++" "-lm" "--start-group" "-lgcc_s" "-lgcc" "-lc" "-lgloss" "--end-group"
// CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtend.o"

// RUN: %clang -### %s -fuse-ld= \
// RUN:   --target=armv6m-none-eabi --rtlib=compiler-rt \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree \
// RUN:   --sysroot=%S/Inputs/basic_arm_gcc_tree/armv6m-none-eabi 2>&1 \
// RUN:   | FileCheck -check-prefix=ARM-BAREMETAL-COMPILER-RT %s

// ARM-BAREMETAL-COMPILER-RT: "{{.*}}crt0.o"
// ARM-BAREMETAL-COMPILER-RT: "{{.*}}clang_rt.crtbegin.o"
// ARM-BAREMETAL-COMPILER-RT: "--start-group" "{{.*}}libclang_rt.builtins.a" "-lc" "-lgloss" "--end-group"
// ARM-BAREMETAL-COMPILER-RT: "{{.*}}clang_rt.crtend.o"

// RUN: %clang -### %s -fuse-ld= \
// RUN:   --target=armv6m-none-eabi --unwindlib=libunwind \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree \
// RUN:   --sysroot=%S/Inputs/basic_arm_gcc_tree/armv6m-none-eabi 2>&1 \
// RUN:   | FileCheck -check-prefix=ARM-BAREMETAL-UNWINDLIB %s

// RUN: %clang -### %s -fuse-ld= \
// RUN:   --target=armv6m-none-eabi --rtlib=compiler-rt --unwindlib=libunwind \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree \
// RUN:   --sysroot=%S/Inputs/basic_arm_gcc_tree/armv6m-none-eabi 2>&1 \
// RUN:   | FileCheck -check-prefix=ARM-BAREMETAL-UNWINDLIB %s

// ARM-BAREMETAL-UNWINDLIB: "{{.*}}crt0.o"
// ARM-BAREMETAL-UNWINDLIB: "{{.*}}clang_rt.crtbegin.o"
// ARM-BAREMETAL-UNWINDLIB: "--start-group" "{{.*}}libclang_rt.builtins.a" "--as-needed" "-lunwind" "--no-as-needed" "-lc" "-lgloss" "--end-group"
// ARM-BAREMETAL-UNWINDLIB: "{{.*}}clang_rt.crtend.o"
