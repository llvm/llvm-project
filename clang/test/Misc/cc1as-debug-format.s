// REQUIRES: x86-registered-target
/// DWARF32 debug info is produced by default, when neither -gdwarf32 nor -gdwarf64 is given.
// RUN: %clang -cc1as -triple x86_64-pc-linux-gnu -filetype obj -debug-info-kind=limited -dwarf-version=4 %s -o %t
// RUN: llvm-dwarfdump -all %t | FileCheck %s --check-prefixes=CHECK,DWARF32
/// -gdwarf64 causes generating DWARF64 debug info.
// RUN: %clang -cc1as -triple x86_64-pc-linux-gnu -filetype obj -gdwarf64 -debug-info-kind=limited -dwarf-version=4 %s -o %t
// RUN: llvm-dwarfdump -all %t | FileCheck %s --check-prefixes=CHECK,DWARF64
/// -gdwarf32 is also handled and produces DWARF32 debug info.
// RUN: %clang -cc1as -triple x86_64-pc-linux-gnu -filetype obj -gdwarf32 -debug-info-kind=limited -dwarf-version=4 %s -o %t
// RUN: llvm-dwarfdump -all %t | FileCheck %s --check-prefixes=CHECK,DWARF32

// CHECK:        .debug_info contents:
// DWARF32-NEXT: format = DWARF32
// DWARF64-NEXT: format = DWARF64

// CHECK:        .debug_line contents:
// CHECK-NEXT:   debug_line[
// CHECK-NEXT:   Line table prologue:
// CHECK-NEXT:     total_length:
// DWARF32-NEXT:     format: DWARF32
// DWARF64-NEXT:     format: DWARF64

.text
  nop
