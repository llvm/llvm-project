// UNSUPPORTED: system-windows

// RUN: %clang -### %s -fuse-ld= \
// RUN:   --target=aarch64-none-elf --rtlib=libgcc \
// RUN:   --gcc-toolchain=%S/Inputs/basic_aarch64_gcc_tree \
// RUN:   --sysroot=%S/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf 2>&1 \
// RUN:   | FileCheck -check-prefix=C-AARCH64-BAREMETAL %s

// C-AARCH64-BAREMETAL: "-cc1" "-triple" "aarch64-unknown-none-elf"
// C-AARCH64-BAREMETAL: "-isysroot" "{{.*}}Inputs/basic_aarch64_gcc_tree/aarch64-none-elf"
// C-AARCH64-BAREMETAL: "-internal-isystem" "{{.*}}Inputs/basic_aarch64_gcc_tree/aarch64-none-elf/include"

// RUN: %clang -### %s -fuse-ld= \
// RUN:   --target=aarch64-none-elf --rtlib=libgcc \
// RUN:   --gcc-toolchain=%S/Inputs/basic_aarch64_gcc_tree \
// RUN:   --sysroot=  2>&1 \
// RUN:   | FileCheck -check-prefix=C-AARCH64-BAREMETAL-NOSYSROOT %s

// C-AARCH64-BAREMETAL-NOSYSROOT: "-cc1" "-triple" "aarch64-unknown-none-elf"
// C-AARCH64-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/../../../../aarch64-none-elf/include"

// RUN: %clangxx -### %s -fuse-ld= \
// RUN:   --target=aarch64-none-elf -stdlib=libstdc++ --rtlib=libgcc \
// RUN:   --gcc-toolchain=%S/Inputs/basic_aarch64_gcc_tree \
// RUN:   --sysroot=%S/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-AARCH64-BAREMETAL %s

// CXX-AARCH64-BAREMETAL: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf/include/c++/8.2.1/aarch64-none-elf"
// CXX-AARCH64-BAREMETAL: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf/include/c++/8.2.1/backward"
// CXX-AARCH64-BAREMETAL: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf/include/c++/8.2.1"
// CXX-AARCH64-BAREMETAL: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf/include"

// RUN: %clangxx -### %s -fuse-ld= \
// RUN:   --target=aarch64-none-elf -stdlib=libstdc++ --rtlib=libgcc \
// RUN:   --gcc-toolchain=%S/Inputs/basic_aarch64_gcc_tree \
// RUN:   --sysroot=  2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-AARCH64-BAREMETAL-NOSYSROOT %s

// CXX-AARCH64-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/../../../../aarch64-none-elf/include/c++/8.2.1/aarch64-none-elf"
// CXX-AARCH64-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/../../../../aarch64-none-elf/include/c++/8.2.1/backward"
// CXX-AARCH64-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/../../../../aarch64-none-elf/include/c++/8.2.1"
// CXX-AARCH64-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/../../../../aarch64-none-elf/include"

// RUN: %clangxx -### %s -fuse-ld= \
// RUN:   --target=aarch64-none-elf -stdlib=libc++ --rtlib=libgcc \
// RUN:   --gcc-toolchain=%S/Inputs/basic_aarch64_gcc_tree \
// RUN:   --sysroot=%S/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-AARCH64-BAREMETAL-LIBCXX %s

// CXX-AARCH64-BAREMETAL-LIBCXX: "-isysroot" "{{.*}}Inputs/basic_aarch64_gcc_tree/aarch64-none-elf"
// CXX-AARCH64-BAREMETAL-LIBCXX: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf/include/c++/v1"
// CXX-AARCH64-BAREMETAL-LIBCXX: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/aarch64-none-elf/include"

// RUN: %clangxx -### %s -fuse-ld= \
// RUN:   --target=aarch64-none-elf -stdlib=libc++ --rtlib=libgcc \
// RUN:   --gcc-toolchain=%S/Inputs/basic_aarch64_gcc_tree \
// RUN:   --sysroot=  2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-AARCH64-BAREMETAL-NOSYSROOT-LIBCXX %s

// CXX-AARCH64-BAREMETAL-NOSYSROOT-LIBCXX: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/../../../../aarch64-none-elf/include/c++/v1"
// CXX-AARCH64-BAREMETAL-NOSYSROOT-LIBCXX: "-internal-isystem" "{{.*}}/Inputs/basic_aarch64_gcc_tree/lib/gcc/aarch64-none-elf/8.2.1/../../../../aarch64-none-elf/include"
