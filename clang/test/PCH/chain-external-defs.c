// REQUIRES: x86-registered-target
// Test with pch.
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -emit-pch -o %t1.pch %S/Inputs/chain-external-defs1.h
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -emit-pch -o %t2.pch %S/Inputs/chain-external-defs2.h -include-pch %t1.pch
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -include-pch %t2.pch -emit-llvm -o %t %s
// RUN: echo FINI >> %t
// RUN: FileCheck -input-file=%t -check-prefix=Z %s
// RUN: FileCheck -input-file=%t -check-prefix=XA %s
// RUN: FileCheck -input-file=%t -check-prefix=YA %s
// RUN: FileCheck -input-file=%t -check-prefix=XB %s
// RUN: FileCheck -input-file=%t -check-prefix=YB %s
// RUN: FileCheck -input-file=%t -check-prefix=AA %s
// RUN: FileCheck -input-file=%t -check-prefix=AB %s
// RUN: FileCheck -input-file=%t -check-prefix=AC %s
// RUN: FileCheck -input-file=%t -check-prefix=S %s

// Z-NOT: @z

// XA: @x = common global i32 0
// XA-NOT: @x = common global i32 0

// YA: @y = common global i32 0
// YA-NOT: @y = common global i32 0

// XB: @x2 = global i32 19
// XB-NOT: @x2 = global i32 19
int x2 = 19;
// YB: @y2 = global i32 18
// YB-NOT: @y2 = global i32 18
int y2 = 18;

// AA: @incomplete_array = common global [1 x i32]
// AA-NOT: @incomplete_array = common global [1 x i32]
// AB: @incomplete_array2 = common global [17 x i32]
// AB-NOT: @incomplete_array2 = common global [17 x i32]
int incomplete_array2[17];
// AC: @incomplete_array3 = common global [1 x i32]
// AC-NOT: @incomplete_array3 = common global [1 x i32]
int incomplete_array3[];

// S: @s = common global %struct.S
// S-NOT: @s = common global %struct.S
struct S {
  int x, y;
};

// Z: FINI
// XA: FINI
// YA: FINI
// XB: FINI
// YB: FINI
// AA: FINI
// AB: FINI
// AC: FINI
// S: FINI
