// RUN: rm -rf %t && mkdir -p %t
// RUN: split-file %s %t

// --- X86_64 MSVCSTL ---
// RUN: %clangxx --target=x86_64-unknown-windows-msvc -### \
// RUN: --sysroot=%t/msvc_tree -fuse-ld=lld %s 2>&1 \
// RUN: | FileCheck --check-prefix=COMPILE_X86_64_MSVCSTL %s
// COMPILE_X86_64_MSVCSTL: clang{{.*}}" "-cc1"
// COMPILE_X86_64_MSVCSTL: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/x86_64-unknown-windows-msvc/c++/msvcstl"
// COMPILE_X86_64_MSVCSTL: "-internal-isystem" "[[SYSROOT]]/include/c++/msvcstl"
// COMPILE_X86_64_MSVCSTL: "-internal-isystem" "[[SYSROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Include{{/|\\\\}}10.0.19041.0{{/|\\\\}}ucrt"
// COMPILE_X86_64_MSVCSTL: "-internal-isystem" "[[SYSROOT]]{{/|\\\\}}VC{{/|\\\\}}Tools{{/|\\\\}}MSVC{{/|\\\\}}14.29.30133{{/|\\\\}}include"
// COMPILE_X86_64_MSVCSTL: lld-link{{.*}}" "-libpath:[[SYSROOT]]/lib/x86_64-unknown-windows-msvc"
// COMPILE_X86_64_MSVCSTL: "-libpath:[[SYSROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Lib{{/|\\\\}}10.0.19041.0{{/|\\\\}}ucrt{{/|\\\\}}x64"
// COMPILE_X86_64_MSVCSTL: "-libpath:[[SYSROOT]]{{/|\\\\}}VC{{/|\\\\}}Tools{{/|\\\\}}MSVC{{/|\\\\}}14.29.30133{{/|\\\\}}lib{{/|\\\\}}x64"

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

// --- AARCH64 MSVCSTL ---
// RUN: %clangxx --target=aarch64-unknown-windows-msvc -### \
// RUN: --sysroot=%t/msvc_tree -fuse-ld=lld %s 2>&1 \
// RUN: | FileCheck --check-prefix=COMPILE_AARCH64_MSVCSTL %s
// COMPILE_AARCH64_MSVCSTL: clang{{.*}}" "-cc1"
// COMPILE_AARCH64_MSVCSTL: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/aarch64-unknown-windows-msvc/c++/msvcstl"
// COMPILE_AARCH64_MSVCSTL: "-internal-isystem" "[[SYSROOT]]/include/c++/msvcstl"
// COMPILE_AARCH64_MSVCSTL: "-internal-isystem" "[[SYSROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Include{{/|\\\\}}10.0.19041.0{{/|\\\\}}ucrt"
// COMPILE_AARCH64_MSVCSTL: "-internal-isystem" "[[SYSROOT]]{{/|\\\\}}VC{{/|\\\\}}Tools{{/|\\\\}}MSVC{{/|\\\\}}14.29.30133{{/|\\\\}}include"
// COMPILE_AARCH64_MSVCSTL: lld-link{{.*}}" "-libpath:[[SYSROOT]]/lib/aarch64-unknown-windows-msvc"
// COMPILE_AARCH64_MSVCSTL: "-libpath:[[SYSROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Lib{{/|\\\\}}10.0.19041.0{{/|\\\\}}ucrt{{/|\\\\}}arm64"
// COMPILE_AARCH64_MSVCSTL: "-libpath:[[SYSROOT]]{{/|\\\\}}VC{{/|\\\\}}Tools{{/|\\\\}}MSVC{{/|\\\\}}14.29.30133{{/|\\\\}}lib{{/|\\\\}}arm64"

// --- ARM64EC MSVCSTL ---
// RUN: %clangxx --target=arm64ec-unknown-windows-msvc -### \
// RUN: --sysroot=%t/msvc_tree -fuse-ld=lld %s 2>&1 \
// RUN: | FileCheck --check-prefix=COMPILE_ARM64EC_MSVCSTL %s
// COMPILE_ARM64EC_MSVCSTL: clang{{.*}}" "-cc1"
// COMPILE_ARM64EC_MSVCSTL: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/arm64ec-unknown-windows-msvc/c++/msvcstl"
// COMPILE_ARM64EC_MSVCSTL: "-internal-isystem" "[[SYSROOT]]/include/c++/msvcstl"
// COMPILE_ARM64EC_MSVCSTL: "-internal-isystem" "[[SYSROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Include{{/|\\\\}}10.0.19041.0{{/|\\\\}}ucrt"
// COMPILE_ARM64EC_MSVCSTL: "-internal-isystem" "[[SYSROOT]]{{/|\\\\}}VC{{/|\\\\}}Tools{{/|\\\\}}MSVC{{/|\\\\}}14.29.30133{{/|\\\\}}include"
// COMPILE_ARM64EC_MSVCSTL: lld-link{{.*}}" "-libpath:[[SYSROOT]]/lib/arm64ec-unknown-windows-msvc" "-libpath:[[SYSROOT]]/lib/aarch64-unknown-windows-msvc"
// COMPILE_ARM64EC_MSVCSTL: "-libpath:[[SYSROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Lib{{/|\\\\}}10.0.19041.0{{/|\\\\}}ucrt{{/|\\\\}}arm64"
// COMPILE_ARM64EC_MSVCSTL: "-libpath:[[SYSROOT]]{{/|\\\\}}VC{{/|\\\\}}Tools{{/|\\\\}}MSVC{{/|\\\\}}14.29.30133{{/|\\\\}}lib{{/|\\\\}}arm64"

// --- LOONGARCH64 MSVCSTL ---
// RUN: %clangxx --target=loongarch64-unknown-windows-msvc -stdlib=msvcstl -### \
// RUN: --sysroot=%t/msvc_tree -fuse-ld=lld %s 2>&1 \
// RUN: | FileCheck --check-prefix=COMPILE_LOONGARCH64_MSVCSTL %s
// COMPILE_LOONGARCH64_MSVCSTL: clang{{.*}}" "-cc1"
// COMPILE_LOONGARCH64_MSVCSTL: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/loongarch64-unknown-windows-msvc/c++/msvcstl"
// COMPILE_LOONGARCH64_MSVCSTL: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/msvcstl"
// COMPILE_LOONGARCH64_MSVCSTL: "-internal-isystem" "[[SYSROOT]]/include/loongarch64-unknown-windows-msvc"
// COMPILE_LOONGARCH64_MSVCSTL: lld-link{{.*}}" "-libpath:[[SYSROOT]]/lib/loongarch64-unknown-windows-msvc" "-libpath:[[SYSROOT]]/lib"

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

// --- AARCH64 print-cxx-stdlib MSVCSTL ---
// RUN: %clangxx --target=aarch64-unknown-windows-msvc --print-cxx-stdlib \
// RUN: --sysroot=%t/msvc_tree %s 2>&1 \
// RUN: | FileCheck --check-prefix=PRINTCXXSTDLIB_AARCH64_MSVCSTL %s
// PRINTCXXSTDLIB_AARCH64_MSVCSTL: msvcstl

// --- X86_64 print-cxx-stdlib LIBCXX ---
// RUN: %clangxx --target=x86_64-unknown-windows-msvc -stdlib=libc++ --print-cxx-stdlib \
// RUN: --sysroot=%t/msvc_tree %s 2>&1 \
// RUN: | FileCheck --check-prefix=PRINTCXXSTDLIB_X86_64_LIBCXX %s
// PRINTCXXSTDLIB_X86_64_LIBCXX: libc++

// --- LOONGARCH64 print-cxx-stdlib LIBSTDCXX ---
// RUN: %clangxx --target=loongarch64-unknown-windows-msvc -stdlib=libstdc++ --print-cxx-stdlib \
// RUN: --sysroot=%t/msvc_tree %s 2>&1 \
// RUN: | FileCheck --check-prefix=PRINTCXXSTDLIB_LOONGARCH64_LIBSTDCXX %s
// PRINTCXXSTDLIB_LOONGARCH64_LIBSTDCXX: libstdc++

// --- X86_64 print-cxx-stdlib-include-dirs MSVCSTL ---
// RUN: %clangxx --target=x86_64-unknown-windows-msvc --print-cxx-stdlib-include-dirs -stdlib=libc++ \
// RUN: --sysroot=%t/msvc_tree 2>&1 \
// RUN: | FileCheck --check-prefix=PRINTCXXSTDLIBINCLUDE_X86_64_LIBCXX %s
// PRINTCXXSTDLIBINCLUDE_X86_64_LIBCXX: [[SYSROOT:[^"]+]]/include/x86_64-unknown-windows-msvc/c++/v1
// PRINTCXXSTDLIBINCLUDE_X86_64_LIBCXX: [[SYSROOT:[^"]+]]/include/c++/v1

// --- AARCH64 print-cxx-stdlib-include-dirs MSVCSTL ---
// RUN: %clangxx --target=aarch64-unknown-windows-msvc --print-cxx-stdlib-include-dirs \
// RUN: --sysroot=%t/msvc_tree 2>&1 \
// RUN: | FileCheck --check-prefix=PRINTCXXSTDLIB_AARCH64_MSVCSTL %s
// PRINTCXXSTDLIBINCLUDE_AARCH64_MSVCSTL: [[SYSROOT:[^"]+]]/include/aarch64-unknown-windows-msvc/c++/msvcstl
// PRINTCXXSTDLIBINCLUDE_AARCH64_MSVCSTL: [[SYSROOT:[^"]+]]/include/c++/msvcstl
// PRINTCXXSTDLIBINCLUDE_AARCH64_MSVCSTL: [[SYSROOT:[^"]+]]/VC/Tools/MSVC/14.29.30133/include

// --- Mock Directory Structure ---
#--- msvc_tree/VC/Tools/MSVC/14.29.30133/include/string
#--- msvc_tree/VC/Tools/MSVC/14.29.30133/lib/x64/msvcrt.lib
#--- msvc_tree/VC/Tools/MSVC/14.29.30133/lib/arm64/msvcrt.lib

#--- msvc_tree/Windows Kits/10/Include/10.0.19041.0/ucrt/assert.h
#--- msvc_tree/Windows Kits/10/Lib/10.0.19041.0/ucrt/x64/ucrt.lib
#--- msvc_tree/Windows Kits/10/Lib/10.0.19041.0/ucrt/arm64/ucrt.lib


#--- msvc_tree/include/c++/msvcstl/string
#--- msvc_tree/include/c++/v1/string
#--- msvc_tree/include/c++/17.0.1/string
#--- msvc_tree/include/x86_64-unknown-windows-msvc/c++/msvcstl/string
#--- msvc_tree/include/aarch64-unknown-windows-msvc/c++/msvcstl/string
#--- msvc_tree/include/loongarch64-unknown-windows-msvc/c++/msvcstl/string
#--- msvc_tree/include/riscv64-unknown-windows-msvc/c++/msvcstl/string
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