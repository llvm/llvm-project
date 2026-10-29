// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -S %s -o - | FileCheck %s -check-prefix=ASM

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm-bc %s -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s -check-prefix=BC

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-obj %s -o %t.o
// RUN: llvm-objdump -t %t.o | FileCheck %s -check-prefix=OBJ

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir-bc %s -o %t.cirbc
// RUN: od -An -c -N4 %t.cirbc | FileCheck %s --check-prefix=CIRBC-MAGIC
// RUN: not grep -q "cir.global" %t.cirbc
// RUN: cir-opt %t.cirbc | FileCheck %s -check-prefix=CIRBC

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -x cir -emit-cir-bc %t.cir -o %t2.cirbc
// RUN: cir-opt %t2.cirbc | FileCheck %s --check-prefix=CIRBC

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %t.cirbc -o - | FileCheck %s --check-prefix=CIRBC

// TODO: Make this test target-independent
// REQUIRES: x86-registered-target

int x = 1;

// BC: @x = {{(dso_local )?}}global i32 1

// ASM: x:
// ASM: .long   1
// ASM: .size   x, 4

// OBJ: .data
// OBJ-SAME: x

// CIRBC-MAGIC: M L 357 R

// CIRBC: cir.global {{.*}}@x = #cir.int<1> : !s32i
