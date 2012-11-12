// RUN: %clang_cc1 -triple i386-apple-darwin9 -emit-llvm %s -o - | FileCheck -check-prefix LSB %s

// CHECK-LSB: @.str = private unnamed_addr constant [8 x i8] c"string0\00"
// CHECK-LSB: @.str1 = linker_private unnamed_addr constant [8 x i8] c"string1\00"
// CHECK-LSB: @.str2 = internal unnamed_addr constant [18 x i16] [i16 104, i16 101, i16 108, i16 108, i16 111, i16 32, i16 8594, i16 32, i16 9731, i16 32, i16 8592, i16 32, i16 119, i16 111, i16 114, i16 108, i16 100, i16 0], align 2
// CHECK-LSB: @.str4 = internal unnamed_addr constant [6 x i16] [i16 116, i16 101, i16 115, i16 116, i16 8482, i16 0], align 2


// RUN: %clang_cc1 -triple powerpc-apple-darwin9 -emit-llvm %s -o - | FileCheck -check-prefix MSB %s

// CHECK-MSB: @.str = private unnamed_addr constant [8 x i8] c"string0\00"
// CHECK-MSB: @.str1 = linker_private unnamed_addr constant [8 x i8] c"string1\00"
// CHECK-MSB: @.str2 = internal unnamed_addr constant [18 x i16] [i16 104, i16 101, i16 108, i16 108, i16 111, i16 32, i16 8594, i16 32, i16 9731, i16 32, i16 8592, i16 32, i16 119, i16 111, i16 114, i16 108, i16 100, i16 0], align 2
// CHECK-MSB: @.str4 = internal unnamed_addr constant [6 x i16] [i16 116, i16 101, i16 115, i16 116, i16 8482, i16 0], align 2

const char *g0 = "string0";
const void *g1 = __builtin___CFStringMakeConstantString("string1");
const void *g2 = __builtin___CFStringMakeConstantString("hello \u2192 \u2603 \u2190 world");
const void *g3 = __builtin___CFStringMakeConstantString("test™");
