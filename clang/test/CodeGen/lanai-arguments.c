// RUN: %clang_cc1 -no-opaque-pointers -triple lanai-unknown-unknown %s -emit-llvm -o - \
// RUN:   | FileCheck %s

// Basic argument/attribute tests for Lanai.

// CHECK: define{{.*}} void @f0(i32 inreg noundef %i, i32 inreg noundef %j, i64 inreg noundef %k)
void f0(int i, long j, long long k) {}

typedef struct {
  int aa;
  int bb;
} s1;
// CHECK: define{{.*}} void @f1(i32 inreg %i.coerce0, i32 inreg %i.coerce1)
void f1(s1 i) {}

typedef struct {
  int cc;
} s2;
// CHECK: define{{.*}} void @f2(%struct.s2* noalias sret(%struct.s2) align 4 %agg.result)
s2 f2(void) {
  s2 foo;
  return foo;
}

typedef struct {
  int cc;
  int dd;
} s3;
// CHECK: define{{.*}} void @f3(%struct.s3* noalias sret(%struct.s3) align 4 %agg.result)
s3 f3(void) {
  s3 foo;
  return foo;
}

// CHECK: define{{.*}} void @f4(i64 inreg noundef %i)
void f4(long long i) {}

// CHECK: define{{.*}} void @f5(i8 inreg noundef %a, i16 inreg noundef %b)
void f5(char a, short b) {}

// CHECK: define{{.*}} void @f6(i8 inreg noundef %a, i16 inreg noundef %b)
void f6(unsigned char a, unsigned short b) {}

enum my_enum {
  ENUM1,
  ENUM2,
  ENUM3,
};
// Enums should be treated as the underlying i32.
// CHECK: define{{.*}} void @f7(i32 inreg noundef %a)
void f7(enum my_enum a) {}

enum my_big_enum {
  ENUM4 = 0xFFFFFFFFFFFFFFFF,
};
// Big enums should be treated as the underlying i64.
// CHECK: define{{.*}} void @f8(i64 inreg noundef %a)
void f8(enum my_big_enum a) {}

union simple_union {
  int a;
  char b;
};
// Unions should be passed inreg.
// CHECK: define{{.*}} void @f9(i32 inreg %s.coerce)
void f9(union simple_union s) {}

typedef struct {
  int b4 : 4;
  int b3 : 3;
  int b8 : 8;
} bitfield1;
// Bitfields should be passed inreg.
// CHECK: define{{.*}} void @f10(i32 inreg %bf1.coerce)
void f10(bitfield1 bf1) {}
