// RUN: %clang_cc1 -emit-llvm -o - -fcuda-is-device -fms-extensions -x hip %s \
// RUN:   -fno-autolink -triple amdgcn-amd-amdhsa \
// RUN:   | FileCheck -check-prefix=DEV %s
// RUN: %clang_cc1 -emit-llvm -o - -fms-extensions -x hip %s -triple \
// RUN:    x86_64-pc-windows-msvc | FileCheck -check-prefix=HOST %s
// RUN: %clang_cc1 -emit-llvm -o - -fcuda-is-device -fms-extensions %s \
// RUN:   -fno-autolink -triple amdgcn-amd-amdhsa \
// RUN:   | FileCheck -check-prefix=DEV %s
// RUN: %clang_cc1 -emit-llvm -o - -fms-extensions %s -triple \
// RUN:    x86_64-pc-windows-msvc | FileCheck -check-prefix=HOST %s

// DEV-NOT: llvm.linker.options
// DEV-NOT: llvm.dependent-libraries
// HOST: lvm.linker.options
// HOST: "/DEFAULTLIB:libcpmt.lib"
// HOST: "/FAILIFMISMATCH:\22myLib_version=9\22"

#pragma comment(lib, "libcpmt")
#pragma detect_mismatch("myLib_version", "9")
