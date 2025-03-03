// UNSUPPORTED: system-windows

// RUN: %clang -### %s -fuse-ld= \
// RUN:   --target=aarch64-none-elf --rtlib=libgcc \
// RUN:   --gcc-toolchain=%S/Inputs/basic_aarch64_gcc_tree \
// RUN:   --sysroot=%S/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf 2>&1 \
// RUN:   | FileCheck -check-prefix=C-AARCH64-BAREMETAL %s

// C-AARCH64-BAREMETAL: "-cc1" "-triple" "aarch64-unknown-none-elf"
// C-AARCH64-BAREMETAL: "-isysroot" "{{.*}}Inputs/basic_aarch64_gcc_tree/aarch64-none-elf"
// C-AARCH64-BAREMETAL: "-internal-isystem" "{{.*}}Inputs/basic_aarch64_gcc_tree/aarch64-none-elf/include"
// C-AARCH64-BAREMETAL: "{{.*}}/ld" "--sysroot={{.*}}/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf"
// C-AARCH64-BAREMETAL: "-Bstatic" "-EL"
// C-AARCH64-BAREMETAL: "{{.*}}/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf/lib/crt0.o"
// C-AARCH64-BAREMETAL: "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/crtbegin.o"
// C-AARCH64-BAREMETAL: "-L{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1"
// C-AARCH64-BAREMETAL: "-L{{.*}}/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf/lib"
// C-AARCH64-BAREMETAL: "--start-group" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "-lc" "-lgloss" "--end-group"
// C-AARCH64-BAREMETAL: "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/crtend.o"

// RUN: %clang -### %s -fuse-ld= \
// RUN:   --target=aarch64-none-elf --rtlib=libgcc \
// RUN:   --gcc-toolchain=%S/Inputs/basic_aarch64_gcc_tree \
// RUN:   --sysroot=  2>&1 \
// RUN:   | FileCheck -check-prefix=C-AARCH64-BAREMETAL-NOSYSROOT %s

// C-AARCH64-BAREMETAL-NOSYSROOT: "-cc1" "-triple" "aarch64-unknown-none-elf"
// C-AARCH64-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/../../../../aarch64-none-elf/include"
// C-AARCH64-BAREMETAL-NOSYSROOT: "{{.*}}/ld" "-Bstatic" "-EL"
// C-AARCH64-BAREMETAL-NOSYSROOT: "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/../../../../aarch64-none-elf/lib/crt0.o"
// C-AARCH64-BAREMETAL-NOSYSROOT: "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/crtbegin.o"
// C-AARCH64-BAREMETAL-NOSYSROOT: "-L{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1"
// C-AARCH64-BAREMETAL-NOSYSROOT: "-L{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/../../../../aarch64-none-elf/lib"
// C-AARCH64-BAREMETAL-NOSYSROOT: "--start-group" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "-lc" "-lgloss" "--end-group"
// C-AARCH64-BAREMETAL-NOSYSROOT: "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/crtend.o"

// RUN: %clangxx -### %s -fuse-ld= \
// RUN:   --target=aarch64-none-elf -stdlib=libstdc++ --rtlib=libgcc \
// RUN:   --gcc-toolchain=%S/Inputs/basic_aarch64_gcc_tree \
// RUN:   --sysroot=%S/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-AARCH64-BAREMETAL %s

// CXX-AARCH64-BAREMETAL: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf/include/c++/8.2.1/aarch64-none-elf"
// CXX-AARCH64-BAREMETAL: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf/include/c++/8.2.1/backward"
// CXX-AARCH64-BAREMETAL: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf/include/c++/8.2.1"
// CXX-AARCH64-BAREMETAL: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf/include"
// CXX-AARCH64-BAREMETAL: "{{.*}}/ld" "--sysroot={{.*}}/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf"
// CXX-AARCH64-BAREMETAL: "-Bstatic" "-EL"
// CXX-AARCH64-BAREMETAL: "{{.*}}/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf/lib/crt0.o"
// CXX-AARCH64-BAREMETAL: "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/crtbegin.o"
// CXX-AARCH64-BAREMETAL: "-L{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1"
// CXX-AARCH64-BAREMETAL: "-L{{.*}}/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf/lib"
// CXX-AARCH64-BAREMETAL: "-lstdc++" "-lm" "--start-group" "-lgcc_s" "-lgcc" "-lc" "-lgloss" "--end-group"
// CXX-AARCH64-BAREMETAL: "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/crtend.o"

// RUN: %clangxx -### %s -fuse-ld= \
// RUN:   --target=aarch64-none-elf -stdlib=libstdc++ --rtlib=libgcc \
// RUN:   --gcc-toolchain=%S/Inputs/basic_aarch64_gcc_tree \
// RUN:   --sysroot=  2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-AARCH64-BAREMETAL-NOSYSROOT %s

// CXX-AARCH64-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/../../../../aarch64-none-elf/include/c++/8.2.1/aarch64-none-elf"
// CXX-AARCH64-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/../../../../aarch64-none-elf/include/c++/8.2.1/backward"
// CXX-AARCH64-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/../../../../aarch64-none-elf/include/c++/8.2.1"
// CXX-AARCH64-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/../../../../aarch64-none-elf/include"
// CXX-AARCH64-BAREMETAL-NOSYSROOT: "{{.*}}/ld" "-Bstatic" "-EL"
// CXX-AARCH64-BAREMETAL-NOSYSROOT: "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/../../../../aarch64-none-elf/lib/crt0.o"
// CXX-AARCH64-BAREMETAL-NOSYSROOT: "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/crtbegin.o"
// CXX-AARCH64-BAREMETAL-NOSYSROOT: "-L{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1"
// CXX-AARCH64-BAREMETAL-NOSYSROOT: "-L{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/../../../../aarch64-none-elf/lib"
// CXX-AARCH64-BAREMETAL-NOSYSROOT: "-lstdc++" "-lm" "--start-group" "-lgcc_s" "-lgcc" "-lc" "-lgloss" "--end-group"
// CXX-AARCH64-BAREMETAL-NOSYSROOT: "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/crtend.o"

// RUN: %clangxx -### %s -fuse-ld= \
// RUN:   --target=aarch64-none-elf -stdlib=libc++ --rtlib=libgcc \
// RUN:   --gcc-toolchain=%S/Inputs/basic_aarch64_gcc_tree \
// RUN:   --sysroot=%S/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-AARCH64-BAREMETAL-LIBCXX %s

// CXX-AARCH64-BAREMETAL-LIBCXX: "-isysroot" "{{.*}}Inputs/basic_aarch64_gcc_tree/aarch64-none-elf"
// CXX-AARCH64-BAREMETAL-LIBCXX: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf/include/c++/v1"
// CXX-AARCH64-BAREMETAL-LIBCXX: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf/include"
// CXX-AARCH64-BAREMETAL-LIBCXX: "{{.*}}/ld" "--sysroot={{.*}}/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf"
// CXX-AARCH64-BAREMETAL-LIBCXX: "-Bstatic" "-EL"
// CXX-AARCH64-BAREMETAL-LIBCXX: "{{.*}}/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf/lib/crt0.o"
// CXX-AARCH64-BAREMETAL-LIBCXX: "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/crtbegin.o"
// CXX-AARCH64-BAREMETAL-LIBCXX: "-L{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1"
// CXX-AARCH64-BAREMETAL-LIBCXX: "-L{{.*}}/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf/lib"
// CXX-AARCH64-BAREMETAL-LIBCXX: "-lc++" "-lm" "--start-group" "-lgcc_s" "-lgcc" "-lc" "-lgloss" "--end-group"
// CXX-AARCH64-BAREMETAL-LIBCXX: "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/crtend.o"

// RUN: %clangxx -### %s -fuse-ld= \
// RUN:   --target=aarch64-none-elf -stdlib=libc++ --rtlib=libgcc \
// RUN:   --gcc-toolchain=%S/Inputs/basic_aarch64_gcc_tree \
// RUN:   --sysroot=  2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-AARCH64-BAREMETAL-NOSYSROOT-LIBCXX %s

// CXX-AARCH64-BAREMETAL-NOSYSROOT-LIBCXX: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/../../../../aarch64-none-elf/include/c++/v1"
// CXX-AARCH64-BAREMETAL-NOSYSROOT-LIBCXX: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/../../../../aarch64-none-elf/include"
// CXX-AARCH64-BAREMETAL-NOSYSROOT-LIBCXX: "{{.*}}/ld" "-Bstatic" "-EL"
// CXX-AARCH64-BAREMETAL-NOSYSROOT-LIBCXX: "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/../../../../aarch64-none-elf/lib/crt0.o"
// CXX-AARCH64-BAREMETAL-NOSYSROOT-LIBCXX: "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/crtbegin.o"
// CXX-AARCH64-BAREMETAL-NOSYSROOT-LIBCXX: "-L{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1"
// CXX-AARCH64-BAREMETAL-NOSYSROOT-LIBCXX: "-L{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/../../../../aarch64-none-elf/lib"
// CXX-AARCH64-BAREMETAL-NOSYSROOT-LIBCXX: "-lc++" "-lm" "--start-group" "-lgcc_s" "-lgcc" "-lc" "-lgloss" "--end-group"
// CXX-AARCH64-BAREMETAL-NOSYSROOT-LIBCXX: "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/crtend.o"

// RUN: %clang -### %s -fuse-ld= \
// RUN:   --target=aarch64-none-elf --rtlib=compiler-rt \
// RUN:   --gcc-toolchain=%S/Inputs/basic_aarch64_gcc_tree \
// RUN:   --sysroot=%S/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf 2>&1 \
// RUN:   | FileCheck -check-prefix=AARCH64-BAREMETAL-COMPILER-RT %s

// AARCH64-BAREMETAL-COMPILER-RT: "{{.*}}crt0.o"
// AARCH64-BAREMETAL-COMPILER-RT: "{{.*}}clang_rt.crtbegin.o"
// AARCH64-BAREMETAL-COMPILER-RT: "--start-group" "{{.*}}libclang_rt.builtins.a" "-lc" "-lgloss" "--end-group"
// AARCH64-BAREMETAL-COMPILER-RT: "{{.*}}clang_rt.crtend.o"

// RUN: %clang -### %s -fuse-ld= \
// RUN:   --target=aarch64-none-elf --unwindlib=libunwind \
// RUN:   --gcc-toolchain=%S/Inputs/basic_aarch64_gcc_tree \
// RUN:   --sysroot=%S/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf 2>&1 \
// RUN:   | FileCheck -check-prefix=AARCH64-BAREMETAL-UNWINDLIB %s

// RUN: %clang -### %s -fuse-ld= \
// RUN:   --target=aarch64-none-elf --rtlib=compiler-rt --unwindlib=libunwind \
// RUN:   --gcc-toolchain=%S/Inputs/basic_aarch64_gcc_tree \
// RUN:   --sysroot=%S/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf 2>&1 \
// RUN:   | FileCheck -check-prefix=AARCH64-BAREMETAL-UNWINDLIB %s

// AARCH64-BAREMETAL-UNWINDLIB: "{{.*}}crt0.o"
// AARCH64-BAREMETAL-UNWINDLIB: "{{.*}}clang_rt.crtbegin.o"
// AARCH64-BAREMETAL-UNWINDLIB: "--start-group" "{{.*}}libclang_rt.builtins.a" "--as-needed" "-lunwind" "--no-as-needed" "-lc" "-lgloss" "--end-group"
// AARCH64-BAREMETAL-UNWINDLIB: "{{.*}}clang_rt.crtend.o"
