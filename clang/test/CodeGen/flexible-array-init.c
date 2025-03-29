// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck %s

struct { int x; int y[]; } a = { 1, 7, 11 };
// CHECK: @a ={{.*}} global { i32, [2 x i32] } { i32 1, [2 x i32] [i32 7, i32 11] }

struct { int y[]; } a1 = { 8, 12 };
// CHECK: @a1 ={{.*}} global { [2 x i32] } { [2 x i32] [i32 8, i32 12] }

struct { int x; int y[]; } b = { 1, { 13, 15 } };
// CHECK: @b ={{.*}} global { i32, [2 x i32] } { i32 1, [2 x i32] [i32 13, i32 15] }

struct { int y[]; } b1 = { { 14, 16 } };
// CHECK: @b1 ={{.*}} global { [2 x i32] } { [2 x i32] [i32 14, i32 16] }

// sizeof(c) == 8, so this global should be at least 8 bytes.
struct { int x; char c; char y[]; } c = { 1, 2, { 13, 15 } };
// CHECK: @c ={{.*}} global { i32, i8, [2 x i8], i8 } { i32 1, i8 2, [2 x i8] c"\0D\0F", i8 0 }

// sizeof(d) == 8, so this global should be at least 8 bytes.
struct __attribute((packed, aligned(4))) { char a; int x; char z[]; } d = { 1, 2, { 13, 15 } };
// CHECK: @d ={{.*}} <{ i8, i32, [2 x i8], i8 }> <{ i8 1, i32 2, [2 x i8] c"\0D\0F", i8 0 }>,

// This global needs 9 bytes to hold all the flexible array members.
struct __attribute((packed, aligned(4))) { char a; int x; char z[]; } e = { 1, 2, { 13, 15, 17, 19 } };
// CHECK: @e ={{.*}} <{ i8, i32, [4 x i8] }> <{ i8 1, i32 2, [4 x i8] c"\0D\0F\11\13" }>

struct { int x; char y[]; } f = { 1, { 13, 15 } };
// CHECK: @f ={{.*}} global <{ i32, [2 x i8] }> <{ i32 1, [2 x i8] c"\0D\0F" }>

struct __attribute((packed)) { short a; char z[]; } g = { 2, { 11, 13, 15 } };
// CHECK: @g ={{.*}} <{ i16, [3 x i8] }> <{ i16 2, [3 x i8] c"\0B\0D\0F" }>,

// Last member is the potential flexible array, unnamed initializer skips it.
struct { int a; union { int b; short x; }; int c; int d; } h = {1, 2, {}, 3};
// CHECK: @h = global %struct.anon{{.*}} { i32 1, %union.anon{{.*}} { i32 2 }, i32 0, i32 3 }
struct { int a; union { int b; short x[0]; }; int c; int d; } h0 = {1, 2, {}, 3};
// CHECK: @h0 = global %struct.anon{{.*}} { i32 1, %union.anon{{.*}} { i32 2 }, i32 0, i32 3 }
struct { int a; union { int b; short x[1]; }; int c; int d; } h1 = {1, 2, {}, 3};
// CHECK: @h1 = global %struct.anon{{.*}} { i32 1, %union.anon{{.*}} { i32 2 }, i32 0, i32 3 }
struct {
  int a;
  union {
    int b;
    struct {
      struct { } __ununsed;
      short x[];
    };
  };
  int c;
  int d;
} hiding = {1, 2, {}, 3};
// CHECK: @hiding = global %struct.anon{{.*}} { i32 1, %union.anon{{.*}} { i32 2 }, i32 0, i32 3 }
struct { int a; union { int b; short x[]; }; int c; int d; } hf = {1, 2, {}, 3};
// CHECK: @hf = global %struct.anon{{.*}} { i32 1, %union.anon{{.*}} { i32 2 }, i32 0, i32 3 }

// First member is the potential flexible array, initialization requires braces.
struct { int a; union { short x; int b; }; int c; int d; } i = {1, 2, {}, 3};
// CHECK: @i = global { i32, { i16, [2 x i8] }, i32, i32 } { i32 1, { i16, [2 x i8] } { i16 2, [2 x i8] zeroinitializer }, i32 0, i32 3 }
struct { int a; union { short x[0]; int b; }; int c; int d; } i0 = {1, {}, 2, 3};
// CHECK: @i0 = global { i32, { [0 x i16], [4 x i8] }, i32, i32 } { i32 1, { [0 x i16], [4 x i8] } zeroinitializer, i32 2, i32 3 }
struct { int a; union { short x[1]; int b; }; int c; int d; } i1 = {1, {2}, {}, 3};
// CHECK: @i1 = global { i32, { [1 x i16], [2 x i8] }, i32, i32 } { i32 1, { [1 x i16], [2 x i8] } { [1 x i16] [i16 2], [2 x i8] zeroinitializer }, i32 0, i32 3 }
struct { int a; union { short x[]; int b; }; int c; int d; } i_f = {4, {}, {}, 6};
// CHECK: @i_f = global { i32, { [0 x i16], [4 x i8] }, i32, i32 } { i32 4, { [0 x i16], [4 x i8] } zeroinitializer, i32 0, i32 6 }

// Named initializers; order doesn't matter.
struct { int a; union { int b; short x; }; int c; int d; } hn = {.a = 1, .x = 2, .c = 3};
// CHECK: @hn = global { i32, { i16, [2 x i8] }, i32, i32 } { i32 1, { i16, [2 x i8] } { i16 2, [2 x i8] zeroinitializer }, i32 3, i32 0 }
struct { int a; union { int b; short x[0]; }; int c; int d; } hn0 = {.a = 1, .x = {2}, .c = 3};
// CHECK: @hn0 = global { i32, { [0 x i16], [4 x i8] }, i32, i32 } { i32 1, { [0 x i16], [4 x i8] } zeroinitializer, i32 3, i32 0 }
struct { int a; union { int b; short x[1]; }; int c; int d; } hn1 = {.a = 1, .x = {2}, .c = 3};
// CHECK: @hn1 = global { i32, { [1 x i16], [2 x i8] }, i32, i32 } { i32 1, { [1 x i16], [2 x i8] } { [1 x i16] [i16 2], [2 x i8] zeroinitializer }, i32 3, i32 0 }

struct { char a[]; } empty_struct = {};
// CHECK: @empty_struct ={{.*}} global %struct.anon{{.*}} zeroinitializer, align 1

struct { char a[]; } empty_struct0 = {0};
// CHECK: @empty_struct0 = global { [1 x i8] } zeroinitializer, align 1

union { struct { int a; char b[]; }; } struct_in_union = {};
// CHECK: @struct_in_union = global %union.anon{{.*}} zeroinitializer, align 4

union { struct { int a; char b[]; }; } struct_in_union0 = {0};
// CHECK: @struct_in_union0 = global %union.anon{{.*}} zeroinitializer, align 4

union { int a; char b[]; } trailing_in_union = {};
// CHECK: @trailing_in_union = global %union.anon{{.*}} zeroinitializer, align 4

union { int a; char b[]; } trailing_in_union0 = {0};
// CHECK: @trailing_in_union0 = global %union.anon{{.*}} zeroinitializer, align 4

union { char a[]; } only_in_union = {};
// CHECK: @only_in_union = global %union.anon{{.*}} zeroinitializer, align 1

union { char a[]; } only_in_union0 = {0};
// CHECK: @only_in_union0 = global { [1 x i8] } zeroinitializer, align 1

union { char a[]; int b; } first_in_union = {};
// CHECK: @first_in_union = global { [0 x i8], [4 x i8] } zeroinitializer, align 4

union { char a[]; int b; } first_in_union0 = {0};
// CHECK: @first_in_union0 = global { [1 x i8], [3 x i8] } zeroinitializer, align 4

union { char a[]; int b; } first_in_union123 = { {1, 2, 3} };
// CHECK: @first_in_union123 = global { [3 x i8], i8 } { [3 x i8] c"\01\02\03", i8 0 }, align 4
