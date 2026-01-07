// UNSUPPORTED: system-windows

// RUN: %clang -### %s -fuse-ld=lld -B%S/Inputs/lld \
// RUN:   --target=armv6m-none-eabi --rtlib=libgcc --unwindlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree \
// RUN:   --sysroot=%S/Inputs/basic_arm_gcc_tree/armv6m-none-eabi 2>&1 \
// RUN:   | FileCheck -check-prefix=LLD-ARM-BAREMETAL %s

// LLD-ARM-BAREMETAL: "-cc1" "-triple" "thumbv6m-unknown-none-eabi"
// LLD-ARM-BAREMETAL: "-isysroot" "{{.*}}Inputs/basic_arm_gcc_tree/armv6m-none-eabi"
// LLD-ARM-BAREMETAL: "-internal-isystem" "{{.*}}Inputs/basic_arm_gcc_tree/armv6m-none-eabi/include"
// LLD-ARM-BAREMETAL: "{{.*}}/Inputs/lld/ld.lld"
// LLD-ARM-BAREMETAL: "-Bstatic" "-m" "armelf" "-EL"
// LLD-ARM-BAREMETAL: "{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/lib/crt0.o"
// LLD-ARM-BAREMETAL: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtbegin.o"
// LLD-ARM-BAREMETAL: "-L{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1"
// LLD-ARM-BAREMETAL: "-L{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/lib"
// LLD-ARM-BAREMETAL: "{{.*}}.o" "--start-group" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "-lc" "-lgloss" "--end-group"
// LLD-ARM-BAREMETAL: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtend.o"

// RUN: %clang -### %s -fuse-ld= \
// RUN:   --target=armv6m-none-eabi --rtlib=libgcc --unwindlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree \
// RUN:   --sysroot=%S/Inputs/basic_arm_gcc_tree/armv6m-none-eabi 2>&1 \
// RUN:   | FileCheck -check-prefix=C-ARM-BAREMETAL %s

// C-ARM-BAREMETAL: "-cc1" "-triple" "thumbv6m-unknown-none-eabi"
// C-ARM-BAREMETAL: "-isysroot" "{{.*}}Inputs/basic_arm_gcc_tree/armv6m-none-eabi"
// C-ARM-BAREMETAL: "-internal-isystem" "{{.*}}Inputs/basic_arm_gcc_tree/armv6m-none-eabi/include"
// C-ARM-BAREMETAL: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../bin/armv6m-none-eabi-ld"
// C-ARM-BAREMETAL: "--sysroot={{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi"
// C-ARM-BAREMETAL: "-Bstatic" "-m" "armelf" "-EL"
// C-ARM-BAREMETAL: "{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/lib/crt0.o"
// C-ARM-BAREMETAL: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtbegin.o"
// C-ARM-BAREMETAL: "-L{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1"
// C-ARM-BAREMETAL: "-L{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/lib"
// C-ARM-BAREMETAL: "{{.*}}.o" "--start-group" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "-lc" "-lgloss" "--end-group"
// C-ARM-BAREMETAL: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtend.o"

// RUN: %clang -### %s -fuse-ld= \
// RUN:   --target=armv6m-none-eabi --rtlib=libgcc --unwindlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree \
// RUN:   --sysroot=  2>&1 \
// RUN:   | FileCheck -check-prefix=C-ARM-BAREMETAL-NOSYSROOT %s

// C-ARM-BAREMETAL-NOSYSROOT: "-cc1" "-triple" "thumbv6m-unknown-none-eabi"
// C-ARM-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/include"
// C-ARM-BAREMETAL-NOSYSROOT: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../bin/armv6m-none-eabi-ld"
// C-ARM-BAREMETAL-NOSYSROOT: "-Bstatic" "-m" "armelf" "-EL"
// C-ARM-BAREMETAL-NOSYSROOT: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/lib/crt0.o"
// C-ARM-BAREMETAL-NOSYSROOT: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtbegin.o"
// C-ARM-BAREMETAL-NOSYSROOT: "-L{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1"
// C-ARM-BAREMETAL-NOSYSROOT: "-L{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/lib"
// C-ARM-BAREMETAL-NOSYSROOT: "{{.*}}.o" "--start-group" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "-lc" "-lgloss" "--end-group"
// C-ARM-BAREMETAL-NOSYSROOT: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtend.o"

// RUN: %clangxx -### %s -fuse-ld= \
// RUN:   --target=armv6m-none-eabi -stdlib=libstdc++ --rtlib=libgcc --unwindlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree \
// RUN:   --sysroot=%S/Inputs/basic_arm_gcc_tree/armv6m-none-eabi 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-ARM-BAREMETAL %s

// CXX-ARM-BAREMETAL: "-isysroot" "{{.*}}Inputs/basic_arm_gcc_tree/armv6m-none-eabi"
// CXX-ARM-BAREMETAL: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/include/c++/8.2.1/armv6m-none-eabi"
// CXX-ARM-BAREMETAL: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/include/c++/8.2.1/backward"
// CXX-ARM-BAREMETAL: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/include/c++/8.2.1"
// CXX-ARM-BAREMETAL: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/include"
// CXX-ARM-BAREMETAL: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../bin/armv6m-none-eabi-ld"
// CXX-ARM-BAREMETAL: "--sysroot={{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi"
// CXX-ARM-BAREMETAL: "-Bstatic" "-m" "armelf" "-EL"
// CXX-ARM-BAREMETAL: "{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/lib/crt0.o"
// CXX-ARM-BAREMETAL: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtbegin.o"
// CXX-ARM-BAREMETAL: "-L{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1"
// CXX-ARM-BAREMETAL: "-L{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/lib"
// CXX-ARM-BAREMETAL: "{{.*}}.o" "-lstdc++" "-lm" "--start-group" "-lgcc_s" "-lgcc" "-lc" "-lgloss" "--end-group"
// CXX-ARM-BAREMETAL: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtend.o"


// RUN: %clangxx -### %s -fuse-ld= \
// RUN:   --target=armv6m-none-eabi -stdlib=libstdc++ --rtlib=libgcc --unwindlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree \
// RUN:   --sysroot=  2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-ARM-BAREMETAL-NOSYSROOT %s

// CXX-ARM-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/include/c++/8.2.1/armv6m-none-eabi"
// CXX-ARM-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/include/c++/8.2.1/backward"
// CXX-ARM-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/include/c++/8.2.1"
// CXX-ARM-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/include"
// CXX-ARM-BAREMETAL-NOSYSROOT: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../bin/armv6m-none-eabi-ld"
// CXX-ARM-BAREMETAL-NOSYSROOT: "-Bstatic" "-m" "armelf" "-EL"
// CXX-ARM-BAREMETAL-NOSYSROOT: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/lib/crt0.o"
// CXX-ARM-BAREMETAL-NOSYSROOT: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtbegin.o"
// CXX-ARM-BAREMETAL-NOSYSROOT: "-L{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1"
// CXX-ARM-BAREMETAL-NOSYSROOT: "-L{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/lib"
// CXX-ARM-BAREMETAL-NOSYSROOT: "{{.*}}.o" "-lstdc++" "-lm" "--start-group" "-lgcc_s" "-lgcc" "-lc" "-lgloss" "--end-group"
// CXX-ARM-BAREMETAL-NOSYSROOT: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtend.o"

// RUN: %clangxx -### %s -fuse-ld= \
// RUN:   --target=armv6m-none-eabi -stdlib=libc++ --rtlib=libgcc --unwindlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree \
// RUN:   --sysroot=%S/Inputs/basic_arm_gcc_tree/armv6m-none-eabi 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-ARM-BAREMETAL-LIBCXX %s

// CXX-ARM-BAREMETAL-LIBCXX: "-isysroot" "{{.*}}Inputs/basic_arm_gcc_tree/armv6m-none-eabi"
// CXX-ARM-BAREMETAL-LIBCXX: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/include/c++/v1"
// CXX-ARM-BAREMETAL-LIBCXX: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/include"
// CXX-ARM-BAREMETAL-LIBCXX: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../bin/armv6m-none-eabi-ld"
// CXX-ARM-BAREMETAL-LIBCXX: "--sysroot={{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi"
// CXX-ARM-BAREMETAL-LIBCXX: "-Bstatic" "-m" "armelf" "-EL"
// CXX-ARM-BAREMETAL-LIBCXX: "{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/lib/crt0.o"
// CXX-ARM-BAREMETAL-LIBCXX: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtbegin.o"
// CXX-ARM-BAREMETAL-LIBCXX: "-L{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1"
// CXX-ARM-BAREMETAL-LIBCXX: "-L{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/lib"
// CXX-ARM-BAREMETAL-LIBCXX: "{{.*}}.o" "-lc++" "-lm" "--start-group" "-lgcc_s" "-lgcc" "-lc" "-lgloss" "--end-group"
// CXX-ARM-BAREMETAL-LIBCXX: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtend.o"

// RUN: %clangxx -### %s -fuse-ld= \
// RUN:   --target=armv6m-none-eabi -stdlib=libc++ --rtlib=libgcc --unwindlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree \
// RUN:   --sysroot=  2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX %s

// CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/include/c++/v1"
// CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/include"
// CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../bin/armv6m-none-eabi-ld"
// CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX: "-Bstatic" "-m" "armelf" "-EL"
// CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/lib/crt0.o"
// CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtbegin.o"
// CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX: "-L{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1"
// CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX: "-L{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/lib"
// CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX: "{{.*}}.o" "-lc++" "-lm" "--start-group" "-lgcc_s" "-lgcc" "-lc" "-lgloss" "--end-group"
// CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX: "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/crtend.o"

// RUN: %clang -### %s -fuse-ld= \
// RUN:   --target=armv6m-none-eabi --rtlib=compiler-rt --unwindlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree \
// RUN:   --sysroot=%S/Inputs/basic_arm_gcc_tree/armv6m-none-eabi 2>&1 \
// RUN:   | FileCheck -check-prefix=ARM-BAREMETAL-COMPILER-RT %s

// ARM-BAREMETAL-COMPILER-RT: "{{.*}}crt0.o"
// ARM-BAREMETAL-COMPILER-RT: "{{.*}}clang_rt.crtbegin.o"
// ARM-BAREMETAL-COMPILER-RT: "--start-group" "{{.*}}libclang_rt.builtins.a" "-lc" "-lgloss" "--end-group"
// ARM-BAREMETAL-COMPILER-RT: "{{.*}}clang_rt.crtend.o"

// RUN: %clang -### %s -fuse-ld= \
// RUN:   --target=armv6m-none-eabi --rtlib=platform --unwindlib=libunwind \
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

// RUN: %clang -static-pie -### %s -fuse-ld= \
// RUN:   --target=armv6m-none-eabi --rtlib=libgcc --unwindlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree \
// RUN:   --sysroot=%S/Inputs/basic_arm_gcc_tree/armv6m-none-eabi 2>&1 \
// RUN:   | FileCheck -check-prefix=C-ARM-STATIC-PIE %s

// C-ARM-STATIC-PIE: "-Bstatic" "-pie" "--no-dynamic-linker" "-z" "text" "-m" "armelf" "-EL"
// C-ARM-STATIC-PIE: "{{.*}}rcrt1.o"
// C-ARM-STATIC-PIE: "{{.*}}crtbeginS.o"
// C-ARM-STATIC-PIE: "--start-group" "-lgcc" "-lgcc_eh" "-lc" "-lgloss" "--end-group"
// C-ARM-STATIC-PIE: "{{.*}}crtendS.o"
