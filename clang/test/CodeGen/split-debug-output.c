// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-unknown-linux -debug-info-kind=limited -split-dwarf-file foo.dwo -split-dwarf-output %t -emit-obj -o - %s | llvm-dwarfdump -debug-info - | FileCheck %s
// RUN: llvm-dwarfdump -debug-info %t | FileCheck %s

int f() { return 0; }

// CHECK: DW_AT_GNU_dwo_name ("foo.dwo")
