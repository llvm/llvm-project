// RUN: %clang_cc1 -debug-info-kind=limited -triple %itanium_abi_triple \
// RUN:   %s -isysroot /CLANG_SYSROOT -emit-llvm -o - \
// RUN:   -debugger-tuning=lldb | FileCheck %s --check-prefix=LLDB
// RUN: %clang_cc1 -debug-info-kind=limited -triple %itanium_abi_triple \
// RUN:   %s -isysroot /CLANG_SYSROOT -emit-llvm -o - \
// RUN:   -debugger-tuning=gdb | FileCheck %s --check-prefix=GDB

void foo() {}

// The sysroot is an LLDB-tuning-specific attribute.

// LLDB: distinct !DICompileUnit({{.*}}sysroot: "/CLANG_SYSROOT"
// GDB: distinct !DICompileUnit(
// GDB-NOT: sysroot: "/CLANG_SYSROOT"

