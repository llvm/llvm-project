// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -S %s -o - | FileCheck %s -check-prefix=ASM

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm-bc %s -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s -check-prefix=BC

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-obj %s -o %t.o
// RUN: llvm-objdump -t %t.o | FileCheck %s -check-prefix=OBJ

// TODO: Make this test target-independent
// REQUIRES: x86-registered-target

int x = 1;

// BC: @x = {{(dso_local )?}}global i32 1

// ASM: x:
// ASM: .long   1
// ASM: .size   x, 4

// OBJ: .data
// OBJ-SAME: x
