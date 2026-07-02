// RUN: rm -rf %t && mkdir -p %t
// RUN: split-file %s %t

// --- X86_64 MSSTL ---
// RUN: %clangxx --target=x86_64-unknown-windows-msvc -### \
// RUN: --sysroot=%t/msvc_tree -fuse-ld=lld %s 2>&1 \
// RUN: | FileCheck --check-prefix=COMPILE_X86_64_MSSTL %s
// COMPILE_X86_64_MSSTL: clang{{.*}}" "-cc1"
// COMPILE_X86_64_MSSTL: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/x86_64-unknown-windows-msvc/c++/msstl"
// COMPILE_X86_64_MSSTL: "-internal-isystem" "[[SYSROOT]]/include/c++/msstl"
// COMPILE_X86_64_MSSTL: "-internal-isystem" "[[SYSROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Include{{/|\\\\}}10.0.19041.0{{/|\\\\}}ucrt"
// COMPILE_X86_64_MSSTL: "-internal-isystem" "[[SYSROOT]]{{/|\\\\}}VC{{/|\\\\}}Tools{{/|\\\\}}MSVC{{/|\\\\}}14.29.30133{{/|\\\\}}include"
// COMPILE_X86_64_MSSTL: lld-link{{.*}}" "-libpath:[[SYSROOT]]/lib/x86_64-unknown-windows-msvc"
// COMPILE_X86_64_MSSTL: "-libpath:[[SYSROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Lib{{/|\\\\}}10.0.19041.0{{/|\\\\}}ucrt{{/|\\\\}}x64"
// COMPILE_X86_64_MSSTL: "-libpath:[[SYSROOT]]{{/|\\\\}}VC{{/|\\\\}}Tools{{/|\\\\}}MSVC{{/|\\\\}}14.29.30133{{/|\\\\}}lib{{/|\\\\}}x64"

// --- X86_64 LIBCXX ---
// RUN: %clangxx --target=x86_64-unknown-windows-msvc -stdlib=libc++ -### \
// RUN: --sysroot=%t/msvc_tree -fuse-ld=lld %s 2>&1 \
// RUN: | FileCheck --check-prefix=COMPILE_X86_64_LIBCXX %s
// COMPILE_X86_64_LIBCXX: clang{{.*}}" "-cc1"
// COMPILE_X86_64_LIBCXX: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/x86_64-unknown-windows-msvc/c++/v1"
// COMPILE_X86_64_LIBCXX: "-internal-isystem" "[[SYSROOT]]/include/c++/v1"
// COMPILE_X86_64_LIBCXX: "-internal-isystem" "[[SYSROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Include{{/|\\\\}}10.0.19041.0{{/|\\\\}}ucrt"
// COMPILE_X86_64_LIBCXX: "-internal-isystem" "[[SYSROOT]]{{/|\\\\}}VC{{/|\\\\}}Tools{{/|\\\\}}MSVC{{/|\\\\}}14.29.30133{{/|\\\\}}include"
// COMPILE_X86_64_LIBCXX: lld-link{{.*}}" "-libpath:[[SYSROOT]]/lib/x86_64-unknown-windows-msvc"
// COMPILE_X86_64_LIBCXX: "-libpath:[[SYSROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Lib{{/|\\\\}}10.0.19041.0{{/|\\\\}}ucrt{{/|\\\\}}x64"
// COMPILE_X86_64_LIBCXX: "-libpath:[[SYSROOT]]{{/|\\\\}}VC{{/|\\\\}}Tools{{/|\\\\}}MSVC{{/|\\\\}}14.29.30133{{/|\\\\}}lib{{/|\\\\}}x64"

// --- X86_64 LIBSTDCXX ---
// RUN: %clangxx -### --target=x86_64-unknown-windows-msvc --stdlib=libstdc++ %s 2>&1 \
// RUN:  -fuse-ld=lld  --sysroot=%t/msvc_tree \
// RUN:   | FileCheck -check-prefix=COMPILE_X86_64_LIBSTDCXX %s
// COMPILE_X86_64_LIBSTDCXX: "-cc1"
// COMPILE_X86_64_LIBSTDCXX: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/17.0.1/x86_64-unknown-windows-msvc"
// COMPILE_X86_64_LIBSTDCXX: "-internal-isystem" "[[SYSROOT]]/include/c++/17.0.1"
// COMPILE_X86_64_LIBSTDCXX: "-internal-isystem" "[[SYSROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Include{{/|\\\\}}10.0.19041.0{{/|\\\\}}ucrt"
// COMPILE_X86_64_LIBSTDCXX: "-internal-isystem" "[[SYSROOT]]{{/|\\\\}}VC{{/|\\\\}}Tools{{/|\\\\}}MSVC{{/|\\\\}}14.29.30133{{/|\\\\}}include"
// COMPILE_X86_64_LIBSTDCXX: lld-link{{.*}}" "-libpath:[[SYSROOT]]/lib/x86_64-unknown-windows-msvc"
// COMPILE_X86_64_LIBSTDCXX: "-libpath:[[SYSROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Lib{{/|\\\\}}10.0.19041.0{{/|\\\\}}ucrt{{/|\\\\}}x64"

// --- AARCH64 MSSTL ---
// RUN: %clangxx --target=aarch64-unknown-windows-msvc -### \
// RUN: --sysroot=%t/msvc_tree -fuse-ld=lld %s 2>&1 \
// RUN: | FileCheck --check-prefix=COMPILE_AARCH64_MSSTL %s
// COMPILE_AARCH64_MSSTL: clang{{.*}}" "-cc1"
// COMPILE_AARCH64_MSSTL: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/aarch64-unknown-windows-msvc/c++/msstl"
// COMPILE_AARCH64_MSSTL: "-internal-isystem" "[[SYSROOT]]/include/c++/msstl"
// COMPILE_AARCH64_MSSTL: "-internal-isystem" "[[SYSROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Include{{/|\\\\}}10.0.19041.0{{/|\\\\}}ucrt"
// COMPILE_AARCH64_MSSTL: "-internal-isystem" "[[SYSROOT]]{{/|\\\\}}VC{{/|\\\\}}Tools{{/|\\\\}}MSVC{{/|\\\\}}14.29.30133{{/|\\\\}}include"
// COMPILE_AARCH64_MSSTL: lld-link{{.*}}" "-libpath:[[SYSROOT]]/lib/aarch64-unknown-windows-msvc"
// COMPILE_AARCH64_MSSTL: "-libpath:[[SYSROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Lib{{/|\\\\}}10.0.19041.0{{/|\\\\}}ucrt{{/|\\\\}}arm64"
// COMPILE_AARCH64_MSSTL: "-libpath:[[SYSROOT]]{{/|\\\\}}VC{{/|\\\\}}Tools{{/|\\\\}}MSVC{{/|\\\\}}14.29.30133{{/|\\\\}}lib{{/|\\\\}}arm64"

// --- ARM64EC MSSTL ---
// RUN: %clangxx --target=arm64ec-unknown-windows-msvc -### \
// RUN: --sysroot=%t/msvc_tree -fuse-ld=lld %s 2>&1 \
// RUN: | FileCheck --check-prefix=COMPILE_ARM64EC_MSSTL %s
// COMPILE_ARM64EC_MSSTL: clang{{.*}}" "-cc1"
// COMPILE_ARM64EC_MSSTL: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/arm64ec-unknown-windows-msvc/c++/msstl"
// COMPILE_ARM64EC_MSSTL: "-internal-isystem" "[[SYSROOT]]/include/c++/msstl"
// COMPILE_ARM64EC_MSSTL: "-internal-isystem" "[[SYSROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Include{{/|\\\\}}10.0.19041.0{{/|\\\\}}ucrt"
// COMPILE_ARM64EC_MSSTL: "-internal-isystem" "[[SYSROOT]]{{/|\\\\}}VC{{/|\\\\}}Tools{{/|\\\\}}MSVC{{/|\\\\}}14.29.30133{{/|\\\\}}include"
// COMPILE_ARM64EC_MSSTL: lld-link{{.*}}" "-libpath:[[SYSROOT]]/lib/arm64ec-unknown-windows-msvc" "-libpath:[[SYSROOT]]/lib/aarch64-unknown-windows-msvc"
// COMPILE_ARM64EC_MSSTL: "-libpath:[[SYSROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Lib{{/|\\\\}}10.0.19041.0{{/|\\\\}}ucrt{{/|\\\\}}arm64"
// COMPILE_ARM64EC_MSSTL: "-libpath:[[SYSROOT]]{{/|\\\\}}VC{{/|\\\\}}Tools{{/|\\\\}}MSVC{{/|\\\\}}14.29.30133{{/|\\\\}}lib{{/|\\\\}}arm64"

// --- LOONGARCH64 MSSTL ---
// RUN: %clangxx --target=loongarch64-unknown-windows-msvc -stdlib=msstl -### \
// RUN: --sysroot=%t/msvc_tree -fuse-ld=lld %s 2>&1 \
// RUN: | FileCheck --check-prefix=COMPILE_LOONGARCH64_MSSTL %s
// COMPILE_LOONGARCH64_MSSTL: clang{{.*}}" "-cc1"
// COMPILE_LOONGARCH64_MSSTL: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/loongarch64-unknown-windows-msvc/c++/msstl"
// COMPILE_LOONGARCH64_MSSTL: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/msstl"
// COMPILE_LOONGARCH64_MSSTL: "-internal-isystem" "[[SYSROOT]]/include/loongarch64-unknown-windows-msvc"
// COMPILE_LOONGARCH64_MSSTL: lld-link{{.*}}" "-libpath:[[SYSROOT]]/lib/loongarch64-unknown-windows-msvc" "-libpath:[[SYSROOT]]/lib"

// --- RISCV64 LIBSTDCXX ---
// RUN: %clangxx --target=riscv64-unknown-windows-msvc -### --stdlib=libstdc++ %s 2>&1 \
// RUN:  --sysroot=%t/msvc_tree -fuse-ld=link \
// RUN:   | FileCheck -check-prefix=COMPILE_RISCV64_LIBSTDCXX_LINK %s
// COMPILE_RISCV64_LIBSTDCXX_LINK: "-cc1"
// COMPILE_RISCV64_LIBSTDCXX_LINK: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/17.0.1/riscv64-unknown-windows-msvc"
// COMPILE_RISCV64_LIBSTDCXX_LINK: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/17.0.1"
// COMPILE_RISCV64_LIBSTDCXX_LINK: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/17.0.1/backward"
// COMPILE_RISCV64_LIBSTDCXX_LINK: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/riscv64-unknown-windows-msvc"
// COMPILE_RISCV64_LIBSTDCXX_LINK: "-internal-isystem" "[[SYSROOT:[^"]+]]/include"
// COMPILE_RISCV64_LIBSTDCXX_LINK: link.exe{{.*}}" "-libpath:[[SYSROOT]]/lib/riscv64-unknown-windows-msvc" "-libpath:[[SYSROOT]]/lib"

// --- Mock Directory Structure ---
#--- msvc_tree/VC/Tools/MSVC/14.29.30133/include/string
#--- msvc_tree/VC/Tools/MSVC/14.29.30133/lib/x64/msvcrt.lib
#--- msvc_tree/VC/Tools/MSVC/14.29.30133/lib/arm64/msvcrt.lib

#--- msvc_tree/Windows Kits/10/Include/10.0.19041.0/ucrt/assert.h
#--- msvc_tree/Windows Kits/10/Lib/10.0.19041.0/ucrt/x64/ucrt.lib
#--- msvc_tree/Windows Kits/10/Lib/10.0.19041.0/ucrt/arm64/ucrt.lib


#--- msvc_tree/include/c++/msstl/string
#--- msvc_tree/include/c++/v1/string
#--- msvc_tree/include/c++/17.0.1/string
#--- msvc_tree/include/x86_64-unknown-windows-msvc/c++/msstl/string
#--- msvc_tree/include/aarch64-unknown-windows-msvc/c++/msstl/string
#--- msvc_tree/include/loongarch64-unknown-windows-msvc/c++/msstl/string
#--- msvc_tree/include/riscv64-unknown-windows-msvc/c++/msstl/string
#--- msvc_tree/include/x86_64-unknown-windows-msvc/c++/v1/string
#--- msvc_tree/include/aarch64-unknown-windows-msvc/c++/v1/string
#--- msvc_tree/include/loongarch64-unknown-windows-msvc/c++/v1/string
#--- msvc_tree/include/riscv64-unknown-windows-msvc/c++/v1/string
#--- msvc_tree/include/x86_64-unknown-windows-msvc/c++/17.0.1/string
#--- msvc_tree/include/aarch64-unknown-windows-msvc/c++/17.0.1/string
#--- msvc_tree/include/loongarch64-unknown-windows-msvc/c++/17.0.1/string
#--- msvc_tree/include/riscv64-unknown-windows-msvc/c++/17.0.1/string
#--- msvc_tree/include/c++/17.0.1/x86_64-unknown-windows-msvc/string
#--- msvc_tree/include/c++/17.0.1/aarch64-unknown-windows-msvc/string
#--- msvc_tree/include/c++/17.0.1/loongarch64-unknown-windows-msvc/string
#--- msvc_tree/include/c++/17.0.1/riscv64-unknown-windows-msvc/string

#--- msvc_tree/include/riscv64-unknown-windows-msvc/empty
#--- msvc_tree/lib/x86_64-unknown-windows-msvc/empty
#--- msvc_tree/lib/aarch64-unknown-windows-msvc/empty
#--- msvc_tree/lib/loongarch64-unknown-windows-msvc/empty
#--- msvc_tree/lib/riscv64-unknown-windows-msvc/empty
#--- msvc_tree/lib/empty

#--- foo.cpp
int main() { return 0; }