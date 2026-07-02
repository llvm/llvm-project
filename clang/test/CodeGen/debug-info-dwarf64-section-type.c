// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-linux-gnu -debug-info-kind=limited \
// RUN:   -dwarf-version=5 -gdwarf64 -emit-obj %s -o %t
// RUN: llvm-readobj --sections %t | FileCheck %s

// CHECK:      Name: .debug_info
// CHECK-NEXT: Type: SHT_DWARF64
// CHECK:      Name: .debug_str
// CHECK-NEXT: Type: SHT_DWARF64

int x;
