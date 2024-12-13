// UNSUPPORTED: system-windows

// RUN: %clang -### %s -fuse-ld= \
// RUN:   --target=armv6m-none-eabi --rtlib=libgcc \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree \
// RUN:   --sysroot=%S/Inputs/basic_arm_gcc_tree/armv6m-none-eabi 2>&1 \
// RUN:   | FileCheck -check-prefix=C-ARM-BAREMETAL %s

// C-ARM-BAREMETAL: "-cc1" "-triple" "thumbv6m-unknown-none-eabi"
// C-ARM-BAREMETAL: "-isysroot" "{{.*}}Inputs/basic_arm_gcc_tree/armv6m-none-eabi"
// C-ARM-BAREMETAL: "-internal-isystem" "{{.*}}Inputs/basic_arm_gcc_tree/armv6m-none-eabi/include"

// RUN: %clang -### %s -fuse-ld= \
// RUN:   --target=armv6m-none-eabi --rtlib=libgcc \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree \
// RUN:   --sysroot=  2>&1 \
// RUN:   | FileCheck -check-prefix=C-ARM-BAREMETAL-NOSYSROOT %s

// C-ARM-BAREMETAL-NOSYSROOT: "-cc1" "-triple" "thumbv6m-unknown-none-eabi"
// C-ARM-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/include"

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

// RUN: %clangxx -### %s -fuse-ld= \
// RUN:   --target=armv6m-none-eabi -stdlib=libstdc++ --rtlib=libgcc \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree \
// RUN:   --sysroot=  2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-ARM-BAREMETAL-NOSYSROOT %s

// CXX-ARM-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/include/c++/8.2.1/armv6m-none-eabi"
// CXX-ARM-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/include/c++/8.2.1/backward"
// CXX-ARM-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/include/c++/8.2.1"
// CXX-ARM-BAREMETAL-NOSYSROOT: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/include"

// RUN: %clangxx -### %s -fuse-ld= \
// RUN:   --target=armv6m-none-eabi -stdlib=libc++ --rtlib=libgcc \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree \
// RUN:   --sysroot=%S/Inputs/basic_arm_gcc_tree/armv6m-none-eabi 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-ARM-BAREMETAL-LIBCXX %s

// CXX-ARM-BAREMETAL-LIBCXX: "-isysroot" "{{.*}}Inputs/basic_arm_gcc_tree/armv6m-none-eabi"
// CXX-ARM-BAREMETAL-LIBCXX: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/include/c++/v1"
// CXX-ARM-BAREMETAL-LIBCXX: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/armv6m-none-eabi/include"

// RUN: %clangxx -### %s -fuse-ld= \
// RUN:   --target=armv6m-none-eabi -stdlib=libc++ --rtlib=libgcc \
// RUN:   --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree \
// RUN:   --sysroot=  2>&1 \
// RUN:   | FileCheck -check-prefix=CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX %s

// CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/include/c++/v1"
// CXX-ARM-BAREMETAL-NOSYSROOT-LIBCXX: "-internal-isystem" "{{.*}}/Inputs/basic_arm_gcc_tree/lib/gcc/armv6m-none-eabi/8.2.1/../../../../armv6m-none-eabi/include
