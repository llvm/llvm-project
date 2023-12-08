// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck %s

struct { int x; int y[]; } a = { 1, 7, 11 };
// CHECK: @a ={{.*}} global { i32, [2 x i32] } { i32 1, [2 x i32] [i32 7, i32 11] }

struct { int x; int y[]; } b = { 1, { 13, 15 } };
// CHECK: @b ={{.*}} global { i32, [2 x i32] } { i32 1, [2 x i32] [i32 13, i32 15] }

// sizeof(c) == 8, so this global should be at least 8 bytes.
struct { int x; char c; char y[]; } c = { 1, 2, { 13, 15 } };
// CHECK: @c ={{.*}} global { i32, i8, [2 x i8] } { i32 1, i8 2, [2 x i8] c"\0D\0F" }

// sizeof(d) == 8, so this global should be at least 8 bytes.
struct __attribute((packed, aligned(4))) { char a; int x; char z[]; } d = { 1, 2, { 13, 15 } };
// CHECK: @d ={{.*}} <{ i8, i32, [2 x i8], i8 }> <{ i8 1, i32 2, [2 x i8] c"\0D\0F", i8 undef }>,

// This global needs 9 bytes to hold all the flexible array members.
struct __attribute((packed, aligned(4))) { char a; int x; char z[]; } e = { 1, 2, { 13, 15, 17, 19 } };
// CHECK: @e ={{.*}} <{ i8, i32, [4 x i8] }> <{ i8 1, i32 2, [4 x i8] c"\0D\0F\11\13" }>

struct { int x; char y[]; } f = { 1, { 13, 15 } };
// CHECK: @f ={{.*}} global <{ i32, [2 x i8] }> <{ i32 1, [2 x i8] c"\0D\0F" }>
