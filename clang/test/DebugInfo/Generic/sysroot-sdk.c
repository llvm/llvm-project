// RUN: %clang_cc1 -debug-info-kind=limited -triple %itanium_abi_triple \
// RUN:   %s -isysroot /CLANG_SYSROOT/MacOSX.sdk -emit-llvm -o - \
// RUN:   -debugger-tuning=lldb | FileCheck %s --check-prefix=LLDB
// RUN: %clang_cc1 -debug-info-kind=limited -triple %itanium_abi_triple \
// RUN:   %s -isysroot /CLANG_SYSROOT/MacOSX.sdk -emit-llvm -o - \
// RUN:   -debugger-tuning=gdb | FileCheck %s --check-prefix=GDB
// RUN: %clang_cc1 -debug-info-kind=limited -triple %itanium_abi_triple \
// RUN:   %s -isysroot /CLANG_SYSROOT/MacOSX.sdk -emit-llvm -o - \
// RUN:   -debugger-tuning=lldb -fno-debug-record-sysroot \
// RUN:   | FileCheck %s --check-prefix=NOSYSROOT

void foo(void) {}

// The sysroot and sdk are LLDB-tuning-specific attributes.

// LLDB: distinct !DICompileUnit({{.*}}sysroot: "/CLANG_SYSROOT/MacOSX.sdk"
// LLDB-SAME:                          sdk: "MacOSX.sdk"
// GDB: distinct !DICompileUnit(
// GDB-NOT: sysroot: "/CLANG_SYSROOT/MacOSX.sdk"
// GDB-NOT: sdk: "MacOSX.sdk"

// -fno-debug-record-sysroot suppresses the sysroot path but keeps the SDK marker.
// NOSYSROOT: distinct !DICompileUnit(
// NOSYSROOT-NOT: sysroot: "/CLANG_SYSROOT/MacOSX.sdk"
// NOSYSROOT-SAME: sdk: "MacOSX.sdk"
