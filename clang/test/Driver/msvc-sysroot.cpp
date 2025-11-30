// RUN: %clangxx --target=x86_64-unknown-windows-msvc -### \
// RUN: --sysroot=%S -fuse-ld=lld %s 2>&1 \
// RUN: | FileCheck --check-prefix=COMPILE_X86_64_MSSTL %s
// COMPILE_X86_64_MSSTL: clang{{.*}}" "-cc1"
// COMPILE_X86_64_MSSTL: "-isysroot" "[[SYSROOT:[^"]+]]"
// COMPILE_X86_64_MSSTL: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/x86_64-unknown-windows-msvc/c++/msstl"
// COMPILE_X86_64_MSSTL: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/msstl"
// COMPILE_X86_64_MSSTL: lld-link{{.*}}" "-libpath:[[SYSROOT:[^"]+]]/lib/x86_64-unknown-windows-msvc" "-libpath:[[SYSROOT:[^"]+]]/lib"

// RUN: %clangxx --target=x86_64-unknown-windows-msvc -### \
// RUN: --sysroot=%S/Inputs/basic_linux_libcxx_tree/usr -stdlib=libc++ -fuse-ld=lld %s 2>&1 \
// RUN: | FileCheck --check-prefix=COMPILE_X86_64_LIBCXX %s
// COMPILE_X86_64_LIBCXX: clang{{.*}}" "-cc1"
// COMPILE_X86_64_LIBCXX: "-isysroot" "[[SYSROOT:[^"]+]]"
// COMPILE_X86_64_LIBCXX: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/x86_64-unknown-windows-msvc/c++/v1"
// COMPILE_X86_64_LIBCXX: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/v1"
// COMPILE_X86_64_LIBCXX: lld-link{{.*}}" "-libpath:[[SYSROOT:[^"]+]]/lib/x86_64-unknown-windows-msvc" "-libpath:[[SYSROOT:[^"]+]]/lib"

// RUN: %clangxx -### --target=x86_64-unknown-windows-msvc --stdlib=libstdc++ %s 2>&1 \
// RUN:  -fuse-ld=lld  --sysroot=%S/Inputs/basic_linux_libstdcxx_libcxxv2_tree/usr \
// RUN:   | FileCheck -check-prefix=COMPILE_X86_64_LIBSTDCXX %s
// COMPILE_X86_64_LIBSTDCXX: "-cc1"
// COMPILE_X86_64_LIBSTDCXX: "-resource-dir" "[[RESOURCE_DIR:[^"]*]]"
// COMPILE_X86_64_LIBSTDCXX: "-isysroot" "[[SYSROOT:[^"]+]]"
// COMPILE_X86_64_LIBSTDCXX: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/4.8/x86_64-unknown-windows-msvc"
// COMPILE_X86_64_LIBSTDCXX: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/4.8"
// COMPILE_X86_64_LIBSTDCXX: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/4.8/backward"
// COMPILE_X86_64_LIBSTDCXX: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// COMPILE_X86_64_LIBSTDCXX: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/x86_64-unknown-windows-msvc"
// COMPILE_X86_64_LIBSTDCXX: "-internal-isystem" "[[SYSROOT:[^"]+]]/include"
// COMPILE_X86_64_LIBSTDCXX: lld-link{{.*}}" "-libpath:[[SYSROOT:[^"]+]]/lib/x86_64-unknown-windows-msvc" "-libpath:[[SYSROOT:[^"]+]]/lib"

// RUN: %clangxx --target=aarch64-unknown-windows-msvc -### \
// RUN: --sysroot=%S -fuse-ld=lld %s 2>&1 \
// RUN: | FileCheck --check-prefix=COMPILE_AARCH64_MSSTL %s
// COMPILE_AARCH64_MSSTL: clang{{.*}}" "-cc1"
// COMPILE_AARCH64_MSSTL: "-isysroot" "[[SYSROOT:[^"]+]]"
// COMPILE_AARCH64_MSSTL: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/aarch64-unknown-windows-msvc/c++/msstl"
// COMPILE_AARCH64_MSSTL: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/msstl"
// COMPILE_AARCH64_MSSTL: lld-link{{.*}}" "-libpath:[[SYSROOT:[^"]+]]/lib/aarch64-unknown-windows-msvc" "-libpath:[[SYSROOT:[^"]+]]/lib"

// RUN: %clangxx --target=arm64ec-unknown-windows-msvc -### \
// RUN: --sysroot=%S -fuse-ld=lld %s 2>&1 \
// RUN: | FileCheck --check-prefix=COMPILE_ARM64EC_MSSTL %s
// COMPILE_ARM64EC_MSSTL: clang{{.*}}" "-cc1"
// COMPILE_ARM64EC_MSSTL: "-isysroot" "[[SYSROOT:[^"]+]]"
// COMPILE_ARM64EC_MSSTL: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/arm64ec-unknown-windows-msvc/c++/msstl"
// COMPILE_ARM64EC_MSSTL: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/msstl"
// COMPILE_ARM64EC_MSSTL: lld-link{{.*}}" "-libpath:[[SYSROOT:[^"]+]]/lib/arm64ec-unknown-windows-msvc" "-libpath:[[SYSROOT:[^"]+]]/lib/aarch64-unknown-windows-msvc" "-libpath:[[SYSROOT:[^"]+]]/lib"

// RUN: %clangxx --target=loongarch64-unknown-windows-msvc -stdlib=msstl -### \
// RUN: --sysroot=%S -fuse-ld=lld %s 2>&1 \
// RUN: | FileCheck --check-prefix=COMPILE_LOONGARCH64_MSSTL %s
// COMPILE_LOONGARCH64_MSSTL: clang{{.*}}" "-cc1"
// COMPILE_LOONGARCH64_MSSTL: "-isysroot" "[[SYSROOT:[^"]+]]"
// COMPILE_LOONGARCH64_MSSTL: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/loongarch64-unknown-windows-msvc/c++/msstl"
// COMPILE_LOONGARCH64_MSSTL: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/msstl"
// COMPILE_LOONGARCH64_MSSTL: lld-link{{.*}}" "-libpath:[[SYSROOT:[^"]+]]/lib/loongarch64-unknown-windows-msvc" "-libpath:[[SYSROOT:[^"]+]]/lib"

// RUN: %clangxx --target=x86_64-unknown-windows-msvc -stdlib=msstl -### \
// RUN: --sysroot=%S %s 2>&1 \
// RUN: | FileCheck --check-prefix=COMPILE_X86_64_MSSTL_LINK %s
// COMPILE_X86_64_MSSTL_LINK: clang{{.*}}" "-cc1"
// COMPILE_X86_64_MSSTL_LINK: "-isysroot" "[[SYSROOT:[^"]+]]"
// COMPILE_X86_64_MSSTL_LINK: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/x86_64-unknown-windows-msvc/c++/msstl"
// COMPILE_X86_64_MSSTL_LINK: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/msstl"
// COMPILE_X86_64_MSSTL_LINK: link.exe{{.*}}" "-libpath:[[SYSROOT:[^"]+]]/lib/x86_64-unknown-windows-msvc" "-libpath:[[SYSROOT:[^"]+]]/lib"

// RUN: %clangxx --target=loongarch64-unknown-windows-msvc -stdlib=libc++ -### \
// RUN: --sysroot=%S/Inputs/basic_linux_libcxx_tree/usr %s 2>&1 \
// RUN: | FileCheck --check-prefix=COMPILE_LOONGARCH64_LIBCXX_LINK %s
// COMPILE_LOONGARCH64_LIBCXX_LINK: clang{{.*}}" "-cc1"
// COMPILE_LOONGARCH64_LIBCXX_LINK: "-isysroot" "[[SYSROOT:[^"]+]]"
// COMPILE_LOONGARCH64_LIBCXX_LINK: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/loongarch64-unknown-windows-msvc/c++/v1"
// COMPILE_LOONGARCH64_LIBCXX_LINK: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/v1"
// COMPILE_LOONGARCH64_LIBCXX_LINK: link.exe{{.*}}" "-libpath:[[SYSROOT:[^"]+]]/lib/loongarch64-unknown-windows-msvc" "-libpath:[[SYSROOT:[^"]+]]/lib"

// RUN: %clangxx --target=riscv64-unknown-windows-msvc -### --stdlib=libstdc++ %s 2>&1 \
// RUN:  --sysroot=%S/Inputs/basic_linux_libstdcxx_libcxxv2_tree/usr \
// RUN:   | FileCheck -check-prefix=COMPILE_RISCV64_LIBSTDCXX_LINK %s
// COMPILE_RISCV64_LIBSTDCXX_LINK: "-cc1"
// COMPILE_RISCV64_LIBSTDCXX_LINK: "-resource-dir" "[[RESOURCE_DIR:[^"]*]]"
// COMPILE_RISCV64_LIBSTDCXX_LINK: "-isysroot" "[[SYSROOT:[^"]+]]"
// COMPILE_RISCV64_LIBSTDCXX_LINK: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/4.8/riscv64-unknown-windows-msvc"
// COMPILE_RISCV64_LIBSTDCXX_LINK: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/4.8"
// COMPILE_RISCV64_LIBSTDCXX_LINK: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/4.8/backward"
// COMPILE_RISCV64_LIBSTDCXX_LINK: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// COMPILE_RISCV64_LIBSTDCXX_LINK: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/riscv64-unknown-windows-msvc"
// COMPILE_RISCV64_LIBSTDCXX_LINK: "-internal-isystem" "[[SYSROOT:[^"]+]]/include"
// COMPILE_RISCV64_LIBSTDCXX_LINK: link.exe{{.*}}" "-libpath:[[SYSROOT:[^"]+]]/lib/riscv64-unknown-windows-msvc" "-libpath:[[SYSROOT:[^"]+]]/lib"
