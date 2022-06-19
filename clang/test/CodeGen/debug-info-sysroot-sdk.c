// RUN: %clang_cc1 -debug-info-kind=limited -triple %itanium_abi_triple \
// RUN:   %s -isysroot /CLANG_SYSROOT/MacOSX.sdk -emit-llvm -o - \
// RUN:   -debugger-tuning=lldb | FileCheck %s --check-prefix=LLDB
// RUN: %clang_cc1 -debug-info-kind=limited -triple %itanium_abi_triple \
// RUN:   %s -isysroot /CLANG_SYSROOT/MacOSX.sdk -emit-llvm -o - \
// RUN:   -debugger-tuning=gdb | FileCheck %s --check-prefix=GDB

void foo(void) {}

// The sysroot and sdk are LLDB-tuning-specific attributes.

// LLDB: distinct !DICompileUnit({{.*}}sysroot: "/CLANG_SYSROOT/MacOSX.sdk",
// LLDB-SAME:                          sdk: "MacOSX.sdk"
// GDB: distinct !DICompileUnit(
// GDB-NOT: sysroot: "/CLANG_SYSROOT/MacOSX.sdk"
// GDB-NOT: sdk: "MacOSX.sdk"
