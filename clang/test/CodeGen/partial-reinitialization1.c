// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -emit-llvm -o - | FileCheck %s

struct P1 {
    struct Q1 {
      char a[6];
      char b[6];
    } q;
};

// CHECK: { [6 x i8] c"foo\00\00\00", [6 x i8] c"\00x\00\00\00\00" }
struct P1 l1 = {
    (struct Q1){ "foo", "bar" },
               .q.b = { "boo" },
               .q.b = { [1] = 'x' }
};

// CHECK: { [6 x i8] c"foo\00\00\00", [6 x i8] c"bxo\00\00\00" }
struct P1 l1a = {
    (struct Q1){ "foo", "bar" },
               .q.b = { "boo" },
               .q.b[1] = 'x'
};


struct P2 { char x[6]; };

// CHECK: { [6 x i8] c"n\00\00\00\00\00" }
struct P2 l2 = {
  .x = { [1] = 'o' },
  .x = { [0] = 'n' }
}; 

struct P3 {
    struct Q3 {
      struct R1 {
         int a, b, c;
      } r1;

      struct R2 {
         int d, e, f;
      } r2;
    } q;
};

// CHECK: @l3 ={{.*}} global %struct.P3 { %struct.Q3 { %struct.R1 { i32 1, i32 2, i32 3 }, %struct.R2 { i32 0, i32 10, i32 0 } } }
struct P3 l3 = {
  (struct Q3){ { 1, 2, 3 }, { 4, 5, 6 } },
                    .q.r2 = { 7, 8, 9 },
                    .q.r2 = { .e = 10 }
};

// This bit is taken from Sema/wchar.c so we can avoid the wchar.h include.      
typedef __WCHAR_TYPE__ wchar_t;                                                  

struct P4 {
    wchar_t x[6];
};

// CHECK: { [6 x i32] [i32 102, i32 111, i32 120, i32 0, i32 0, i32 0] }
struct P4 l4 = { { L"foo" }, .x[2] = L'x' };

struct P5 {
  int x;
  struct Q5 {
    int a, b, c;
  } q;
  int y;
};

// A three-pass test
// CHECK: @l5 ={{.*}} global %struct.P5 { i32 1, %struct.Q5 { i32 6, i32 9, i32 8 }, i32 5 }
struct P5 l5 = { 1, { 2, 3, 4 }, 5,
                 .q = { 6, 7, 8 },
                 .q.b = 9 };
