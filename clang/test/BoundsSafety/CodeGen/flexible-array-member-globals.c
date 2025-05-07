
// RUN: %clang_cc1 -O0 -triple x86_64 -fbounds-safety -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -O0 -triple x86_64 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm %s -o - | FileCheck %s

#include <ptrcheck.h>

struct flex {
	unsigned char count;
	unsigned elems[__counted_by(count)];
};

// CHECK: @f0 = global { i8, [3 x i8], [0 x i32] } zeroinitializer, align 4
struct flex f0 = { };

// CHECK: @f1 = global { i8, [3 x i8], [0 x i32] } zeroinitializer, align 4
struct flex f1 = { 0 };

// CHECK: @f2 = global { i8, [3 x i8], [0 x i32] } zeroinitializer, align 4
struct flex f2 = { .count = 0 };

// CHECK: @f3 = global { i8, [3 x i8], [0 x i32] } zeroinitializer, align 4
struct flex f3 = { 0, {} };

// CHECK: @f4 = global { i8, [3 x i8], [0 x i32] } zeroinitializer, align 4
struct flex f4 = { .count = 0, {} };

// CHECK: @f5 = global { i8, [3 x i8], [0 x i32] } zeroinitializer, align 4
struct flex f5 = { .count = 0, .elems = {} };

// CHECK: @f6 = global { i8, [3 x i8], [1 x i32] } { i8 1, [3 x i8] zeroinitializer, [1 x i32] [i32 1] }, align 4
struct flex f6 = { .count = 1, { 1 } };

// CHECK: @f7 = global { i8, [3 x i8], [1 x i32] } { i8 1, [3 x i8] zeroinitializer, [1 x i32] [i32 1] }, align 4
struct flex f7 = { 1, { 1 } };

// CHECK: @f8 = global { i8, [3 x i8], [1 x i32] } { i8 1, [3 x i8] zeroinitializer, [1 x i32] [i32 1] }, align 4
struct flex f8 = { .count = 1, { 1 } };

// CHECK: @f9 = global { i8, [3 x i8], [1 x i32] } { i8 1, [3 x i8] zeroinitializer, [1 x i32] [i32 1] }, align 4
struct flex f9 = { .count = 1, .elems = { 1 } };

// CHECK: @f10 = global { i8, [3 x i8], [3 x i32] } { i8 3, [3 x i8] zeroinitializer, [3 x i32] [i32 4, i32 1, i32 2] }, align 4
struct flex f10 = { .count = 3, { 4, 1, 2 } };

// CHECK: @f11 = global { i8, [3 x i8], [3 x i32] } { i8 3, [3 x i8] zeroinitializer, [3 x i32] [i32 4, i32 1, i32 2] }, align 4
struct flex f11 = { .count = 3, { 4, 1, [2] = 2 } };

// CHECK: @f12 = global { i8, [3 x i8], [3 x i32] } { i8 3, [3 x i8] zeroinitializer, [3 x i32] [i32 4, i32 0, i32 3] }, align 4
struct flex f12 = { .count = 3, { 4, [2] = 3 } };

// CHECK: @f13 = global { i8, [3 x i8], [3 x i32] } { i8 3, [3 x i8] zeroinitializer, [3 x i32] [i32 0, i32 0, i32 3] }, align 4
struct flex f13 = { .count = 3, { [2] = 3 } };
