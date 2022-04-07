// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-macosx10.7.0 -emit-llvm -o - %s | FileCheck %s

// PR4610
#pragma pack(4)
struct ref {
        struct ref *next;
} refs;

// PR13580
struct S
{
  char a[3];
#pragma pack(1)
  struct T
  {
    char b;
    int c;
  } d;
#pragma pack()
  struct T2
  {
    char b;
    int c;
  } d2;
} ss;

struct S3
{
  char a[3];
#pragma pack(push, 2)
  struct T3
  {
    char b;
    int c;
  } d;
#pragma pack(pop)
  struct T32
  {
    char b;
    int c;
  } e;
} s3;

struct S4
{
  char a[3];
#pragma align=packed
  struct T4
  {
    char b;
    int c;
  } d;
  int e;
} s4;

// CHECK: [[struct_ref:%[a-zA-Z0-9_.]+]] = type { [[struct_ref]]* }
// CHECK: [[struct_S:%[a-zA-Z0-9_.]+]] = type { [3 x i8], [[struct_T:%[a-zA-Z0-9_.]+]], [[struct_T2:%[a-zA-Z0-9_.]+]] }
// CHECK: [[struct_T]] = type <{ i8, i32 }>
// CHECK: [[struct_T2]] = type { i8, i32 }

// CHECK: %struct.S3 = type { [3 x i8], i8, %struct.T3, %struct.T32 }
// CHECK: %struct.T3 = type <{ i8, i8, i32 }>
// CHECK: %struct.T32 = type { i8, i32 }
// CHECK: %struct.S4 = type { [3 x i8], %struct.T4, i32 }
// CHECK: %struct.T4 = type <{ i8, i32 }>

// CHECK: @refs ={{.*}} global [[struct_ref]]
// CHECK: @ss ={{.*}} global [[struct_S]]
