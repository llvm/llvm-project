/// Verify that unnamed symbols are not added as function entry points. Such
/// symbols are used by relocations in debugging sections.

// clang-format off

// RUN: %clang %cflags -g -Wl,-q -o %t %s

/// Verify that the binary indeed contains a fake label ".L0 " at _start.
// RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=CHECK-ELF
// CHECK-ELF-DAG: [[#%x,START:]] {{.*}} FUNC GLOBAL DEFAULT [[#%d,SECTION:]] _start{{$}}
// CHECK-ELF-DAG: [[#%x,START]] {{.*}} NOTYPE LOCAL DEFAULT [[#SECTION]] .L0 {{$}}

/// Verify that BOLT did not create an extra entry point for the fake label.
// RUN: llvm-bolt -o %t.bolt %t --print-cfg | FileCheck %s
// CHECK: Binary Function "_start" after building cfg {
// CHECK:  IsMultiEntry: 0

void _start() {}
