// RUN: %clang_cc1 -triple arc-unknown-unknown %s -emit-llvm -o - \
// RUN:   | FileCheck %s

// Basic argument tests for ARC.

// CHECK: define{{.*}} void @f0(i32 inreg noundef %i, i32 inreg noundef %j, i64 inreg noundef %k)
void f0(int i, long j, long long k) {}

typedef struct {
  int aa;
  int bb;
} s1;
// CHECK: define{{.*}} void @f1(i32 inreg %i.coerce0, i32 inreg %i.coerce1)
void f1(s1 i) {}

typedef struct {
  char aa; char bb; char cc; char dd;
} cs1;
// CHECK: define{{.*}} void @cf1(i32 inreg %i.coerce)
void cf1(cs1 i) {}

typedef struct {
  int cc;
} s2;
// CHECK: define{{.*}} void @f2(%struct.s2* noalias sret(%struct.s2) align 4 %agg.result)
s2 f2() {
  s2 foo;
  return foo;
}

typedef struct {
  int cc;
  int dd;
} s3;
// CHECK: define{{.*}} void @f3(%struct.s3* noalias sret(%struct.s3) align 4 %agg.result)
s3 f3() {
  s3 foo;
  return foo;
}

// CHECK: define{{.*}} void @f4(i64 inreg noundef %i)
void f4(long long i) {}

// CHECK: define{{.*}} void @f5(i8 inreg noundef signext %a, i16 inreg noundef signext %b)
void f5(signed char a, short b) {}

// CHECK: define{{.*}} void @f6(i8 inreg noundef zeroext %a, i16 inreg noundef zeroext %b)
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

// CHECK: define{{.*}} inreg { float, float } @cplx1(float inreg noundef %r)
_Complex float cplx1(float r) {
  return r + 2.0fi;
}

// CHECK: define{{.*}} inreg { double, double } @cplx2(double inreg noundef %r)
_Complex double cplx2(double r) {
  return r + 2.0i;
}

// CHECK: define{{.*}} inreg { i32, i32 } @cplx3(i32 inreg noundef %r)
_Complex int cplx3(int r) {
  return r + 2i;
}

// CHECK: define{{.*}} inreg { i64, i64 } @cplx4(i64 inreg noundef %r)
_Complex long long cplx4(long long r) {
  return r + 2i;
}

// CHECK: define{{.*}} inreg { i8, i8 } @cplx6(i8 inreg noundef signext %r)
_Complex signed char cplx6(signed char r) {
  return r + 2i;
}

// CHECK: define{{.*}} inreg { i16, i16 } @cplx7(i16 inreg noundef signext %r)
_Complex short cplx7(short r) {
  return r + 2i;
}

typedef struct {
  int aa; int bb;
} s8;

typedef struct {
  int aa; int bb; int cc; int dd;
} s16;

// Use 16-byte struct 2 times, gets 8 registers.
void st2(s16 a, s16 b) {}
// CHECK: define{{.*}} void @st2(i32 inreg %a.coerce0, i32 inreg %a.coerce1, i32 inreg %a.coerce2, i32 inreg %a.coerce3, i32 inreg %b.coerce0, i32 inreg %b.coerce1, i32 inreg %b.coerce2, i32 inreg %b.coerce3)

// Use 8-byte struct 3 times, gets 8 registers, 1 byval struct argument.
void st3(s16 a, s16 b, s16 c) {}
// CHECK: define{{.*}} void @st3(i32 inreg %a.coerce0, i32 inreg %a.coerce1, i32 inreg %a.coerce2, i32 inreg %a.coerce3, i32 inreg %b.coerce0, i32 inreg %b.coerce1, i32 inreg %b.coerce2, i32 inreg %b.coerce3, { i32, i32, i32, i32 } %c.coerce)

// 1 sret + 1 i32 + 2*(i32 coerce) + 4*(i32 coerce) + 1 byval
s16 st4(int x, s8 a, s16 b, s16 c) { return b; }
// CHECK: define{{.*}} void @st4(%struct.s16* noalias sret(%struct.s16) align 4 %agg.result, i32 inreg noundef %x, i32 inreg %a.coerce0, i32 inreg %a.coerce1, i32 inreg %b.coerce0, i32 inreg %b.coerce1, i32 inreg %b.coerce2, i32 inreg %b.coerce3, { i32, i32, i32, i32 } %c.coerce)

// 1 sret + 2*(i32 coerce) + 4*(i32 coerce) + 4*(i32 coerce)
s16 st5(s8 a, s16 b, s16 c) { return b; }
// CHECK: define{{.*}} void @st5(%struct.s16* noalias sret(%struct.s16) align 4 %agg.result, i32 inreg %a.coerce0, i32 inreg %a.coerce1, i32 inreg %b.coerce0, i32 inreg %b.coerce1, i32 inreg %b.coerce2, i32 inreg %b.coerce3, { i32, i32, i32, i32 } %c.coerce)
