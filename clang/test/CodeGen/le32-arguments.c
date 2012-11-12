// RUN: %clang_cc1 -triple le32-unknown-nacl %s -emit-llvm -o - | FileCheck %s

// Basic argument/attribute tests for le32/PNaCl

// CHECK: define void @f0(i32 %i, i32 %j, double %k)
void f0(int i, long j, double k) {}

typedef struct {
  int aa;
  int bb;
} s1;
// Structs should be passed byval and not split up
// CHECK: define void @f1(%struct.s1* byval %i)
void f1(s1 i) {}

typedef struct {
  int cc;
} s2;
// Structs should be returned sret and not simplified by the frontend
// CHECK: define void @f2(%struct.s2* noalias sret %agg.result)
s2 f2() {
  s2 foo;
  return foo;
}

// CHECK: define void @f3(i64 %i)
void f3(long long i) {}

// i8/i16 should be signext, i32 and higher should not
// CHECK: define void @f4(i8 signext %a, i16 signext %b)
void f4(char a, short b) {}

// CHECK: define void @f5(i8 zeroext %a, i16 zeroext %b)
void f5(unsigned char a, unsigned short b) {}


enum my_enum {
  ENUM1,
  ENUM2,
  ENUM3,
};
// Enums should be treated as the underlying i32
// CHECK: define void @f6(i32 %a)
void f6(enum my_enum a) {}

union simple_union {
  int a;
  char b;
};
// Unions should be passed as byval structs
// CHECK: define void @f7(%union.simple_union* byval %s)
void f7(union simple_union s) {}

typedef struct {
  int b4 : 4;
  int b3 : 3;
  int b8 : 8;
} bitfield1;
// Bitfields should be passed as byval structs
// CHECK: define void @f8(%struct.bitfield1* byval %bf1)
void f8(bitfield1 bf1) {}
