// RUN: %clangxx -print-cxx-stdlib -stdlib=libc++ \
// RUN:   --target=x86_64-unknown-linux-gnu | FileCheck %s --check-prefix=LIBCXX
// RUN: %clangxx --print-cxx-stdlib -stdlib=libstdc++ \
// RUN:   --target=x86_64-unknown-linux-gnu | FileCheck %s --check-prefix=LIBSTDCXX
// RUN: %clangxx --print-cxx-stdlib --target=x86_64-pc-windows-msvc \
// RUN:   | FileCheck %s --check-prefix=MSVC-STL
// RUN: %clangxx --print-cxx-stdlib -stdlib=libc++ \
// RUN:   --target=x86_64-pc-windows-msvc | FileCheck %s --check-prefix=LIBCXX

// RUN: mkdir -p %t/bin
// RUN: mkdir -p %t/include/c++/v1
// RUN: %clangxx -print-cxx-stdlib-include-dirs -stdlib=libc++ \
// RUN:   --target=x86_64-unknown-linux-gnu -ccc-install-dir %t/bin \
// RUN:   | FileCheck %s --check-prefix=LIBCXX-INCLUDES

// RUN: %clangxx --print-cxx-stdlib-include-dirs -stdlib++-isystem /tmp/foo \
// RUN:   -stdlib++-isystem /tmp/bar --target=x86_64-unknown-linux-gnu \
// RUN:   | FileCheck %s --check-prefix=STDLIBXX-ISYSTEM
// RUN: %clangxx --print-cxx-stdlib-include-dirs -stdlib++-isystem /tmp/foo \
// RUN:   -stdlib++-isystem /tmp/bar -nostdinc++ \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   | FileCheck %s --check-prefix=NO-INCLUDES --allow-empty

// RUN: %clangxx --print-cxx-stdlib-include-dirs -stdlib=libstdc++ \
// RUN:   --gcc-toolchain=%S/Inputs/gcc_version_parsing_rt_libs \
// RUN:   --target=x86_64-redhat-linux \
// RUN:   | FileCheck %s --check-prefix=LIBSTDCXX-INCLUDES
// RUN: %clangxx --print-cxx-stdlib-include-dirs \
// RUN:   --target=x86_64-pc-windows-msvc \
// RUN:   -Xmicrosoft-visualc-tools-root %t/VC/Tools/MSVC/27.1828.18284 \
// RUN:   | FileCheck %s --check-prefix=MSVC-INCLUDES
// RUN: %clangxx --print-cxx-stdlib-include-dirs \
// RUN:   --target=x86_64-pc-windows-msvc \
// RUN:   -Xmicrosoft-visualc-tools-root %t/VC/Tools/MSVC/27.1828.18284 \
// RUN:   -nostdinc++ \
// RUN:   | FileCheck %s --check-prefix=NO-INCLUDES --allow-empty
// RUN: %clangxx --print-cxx-stdlib-include-dirs -stdlib=libc++ \
// RUN:   --target=x86_64-pc-windows-msvc \
// RUN:   -Xmicrosoft-visualc-tools-root %t/VC/Tools/MSVC/27.1828.18284 \
// RUN:   | FileCheck %s --check-prefix=NO-INCLUDES --allow-empty

// LIBCXX: libc++
// LIBSTDCXX: libstdc++
// MSVC-STL: msvcstl

// LIBCXX-INCLUDES: {{.*}}{{/|\\}}include{{/|\\}}c++{{/|\\}}v1

// STDLIBXX-ISYSTEM: /tmp/foo
// STDLIBXX-ISYSTEM-NEXT: /tmp/bar

// NO-INCLUDES-NOT: {{.}}

// LIBSTDCXX-INCLUDES: {{.*}}gcc_version_parsing_rt_libs{{/|\\}}lib{{/|\\}}gcc{{/|\\}}x86_64-redhat-linux{{/|\\}}10.2.0{{/|\\}}..{{/|\\}}..{{/|\\}}..{{/|\\}}gcc{{/|\\}}x86_64-redhat-linux{{/|\\}}10.2.0{{/|\\}}include{{/|\\}}c++

// MSVC-INCLUDES: {{.*}}VC{{/|\\}}Tools{{/|\\}}MSVC{{/|\\}}27.1828.18284{{/|\\}}include
// MSVC-INCLUDES-NOT: atlmfc
