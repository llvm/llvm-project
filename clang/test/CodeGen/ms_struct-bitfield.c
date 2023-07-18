// RUN: %clang_cc1 -emit-llvm -o - -triple x86_64-apple-darwin9 %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -o - -triple thumbv7-apple-ios -target-abi apcs-gnu %s | FileCheck %s -check-prefix=CHECK-ARM

// rdar://8823265

// Note that we're declaring global variables with these types,
// triggering both Sema and IRGen struct layout.

#define ATTR __attribute__((__ms_struct__))

struct
{
   char foo;
   long : 0;
   char bar;
} ATTR t1;
int s1 = sizeof(t1);
// CHECK: @s1 ={{.*}} global i32 2
// CHECK-ARM: @s1 ={{.*}} global i32 2

struct
{
   char foo;
   long : 0;
   char : 0;
   int : 0;
   char bar;
} ATTR t2;
int s2 = sizeof(t2);
// CHECK: @s2 ={{.*}} global i32 2
// CHECK-ARM: @s2 ={{.*}} global i32 2

struct
{
   char foo;
   long : 0;
   char : 0;
   int : 0;
   char bar;
   long : 0;
   char : 0;
} ATTR t3;
int s3 = sizeof(t3);
// CHECK: @s3 ={{.*}} global i32 2
// CHECK-ARM: @s3 ={{.*}} global i32 2

struct
{
   long : 0;
   char bar;
} ATTR t4;
int s4 = sizeof(t4);
// CHECK: @s4 ={{.*}} global i32 1
// CHECK-ARM: @s4 ={{.*}} global i32 1

struct
{
   long : 0;
   long : 0;
   char : 0;
   char bar;
} ATTR t5;
int s5 = sizeof(t5);
// CHECK: @s5 ={{.*}} global i32 1
// CHECK-ARM: @s5 ={{.*}} global i32 1

struct
{
   long : 0;
   long : 0;
   char : 0;
   char bar;
} ATTR t6;
int s6 = sizeof(t6);
// CHECK: @s6 ={{.*}} global i32 1
// CHECK-ARM: @s6 ={{.*}} global i32 1

struct
{
   char foo;
   long : 0;
   int : 0;
   char bar;
   char bar1;
   long : 0;
   char bar2;
   char bar3;
   char : 0;
   char bar4;
   char bar5;
   char : 0;
   char bar6;
   char bar7;
} ATTR t7;
int s7 = sizeof(t7);
// CHECK: @s7 ={{.*}} global i32 9
// CHECK-ARM: @s7 ={{.*}} global i32 9

struct
{
   long : 0;
   long : 0;
   char : 0;
} ATTR t8;
int s8 = sizeof(t8);
// CHECK: @s8 ={{.*}} global i32 0
// CHECK-ARM: @s8 ={{.*}} global i32 0

struct
{
   char foo;
   long : 0;
   int : 0;
   char bar;
   char bar1;
   long : 0;
   char bar2;
   char bar3;
   char : 0;
   char bar4;
   char bar5;
   char : 0;
   char bar6;
   char bar7;
   int  i1;
   char : 0;
   long : 0;
   char :4;
   char bar8;
   char : 0;
   char bar9;
   char bar10;
   int  i2;
   char : 0;
   long : 0;
   char :4;
} ATTR t9;
int s9 = sizeof(t9);
// CHECK: @s9 ={{.*}} global i32 28
// CHECK-ARM: @s9 ={{.*}} global i32 28

struct
{
   char foo: 8;
   long : 0;
   char bar;
} ATTR t10;
int s10 = sizeof(t10);
// CHECK: @s10 ={{.*}} global i32 16
// CHECK-ARM: @s10 ={{.*}} global i32 8

// rdar://16041826 - ensure that ms_structs work correctly on a
// !useBitFieldTypeAlignment() target
struct {
  unsigned int a : 31;
  unsigned int b : 2;
  unsigned int c;
} ATTR t11;
int s11 = sizeof(t11);
// CHECK: @s11 ={{.*}} global i32 12
// CHECK-ARM: @s11 ={{.*}} global i32 12

struct {
  unsigned char a : 3;
  unsigned char b : 4;
  unsigned short c : 6;
} ATTR t12;
int s12 = sizeof(t12);
// CHECK: @s12 ={{.*}} global i32 4
// CHECK-ARM: @s12 ={{.*}} global i32 4

struct {
  unsigned char a : 3;
  unsigned char b : 4;
  __attribute__((packed)) unsigned short c : 6;
} ATTR t13;
int s13 = sizeof(t13);
// CHECK: @s13 ={{.*}} global i32 4
// CHECK-ARM: @s13 ={{.*}} global i32 4
