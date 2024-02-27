// RUN: mkdir -p %t/src
// RUN: cp %s %t/src/debug-info-debug-prefix-map.c

// RUN: mkdir -p %t/out
// RUN: cd %t/out
// RUN: %clang_cc1 -debug-info-kind=limited -triple %itanium_abi_triple \
// RUN:   -fdebug-prefix-map="%t/=./" %t/src/debug-info-debug-prefix-map.c \
// RUN:   -emit-llvm -o - | FileCheck %s

void foo(void) {}

// Compile unit filename is transformed from absolute path %t/src... to
// a relative path ./src... But it should not be relative to directory "./out".

// CHECK: = distinct !DICompileUnit({{.*}}file: ![[#CUFILE:]]
// CHECK: ![[#CUFILE]] = !DIFile(
// CHECK-NOT:    directory: "./out"
// CHECK-SAME:   filename: "./src{{[^"]+}}"
// CHECK-NOT:    directory: "./out"
// CHECK-SAME: )
