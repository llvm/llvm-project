// RUN: %clang_cc1 -triple x86_64-unknown-nacl -emit-llvm -o - %s| FileCheck %s
#include <stdarg.h>
// Test for x86-64 structure representation (instead of pnacl representation),
// in particular for unions. Also crib a few tests from x86 Linux.

union PP_VarValue {
  int as_int;
  double as_double;
  long long as_i64;
};

struct PP_Var {
  int type;
  int padding;
  union PP_VarValue value;
};

// CHECK: define { i64, i64 } @f0()
struct PP_Var f0() {
  struct PP_Var result = { 0, 0, 0 };
  return result;
}

// CHECK: define void @f1(i64 %p1.coerce0, i64 %p1.coerce1)
void f1(struct PP_Var p1) { while(1) {} }

// long doubles are 64 bits on NaCl
// CHECK: define double @f5()
long double f5(void) {
  return 0;
}

// CHECK: define void @f6(i8 signext %a0, i16 signext %a1, i32 %a2, i64 %a3, i8* %a4)
void f6(char a0, short a1, int a2, long long a3, void *a4) {
}

// CHECK: define i64 @f8_1()
// CHECK: define void @f8_2(i64 %a0.coerce)
union u8 {
  long double a;
  int b;
};
union u8 f8_1() { while (1) {} }
void f8_2(union u8 a0) {}

// CHECK: define i64 @f9()
struct s9 { int a; int b; int : 0; } f9(void) { while (1) {} }

// CHECK: define void @f10(i64 %a0.coerce)
struct s10 { int a; int b; int : 0; };
void f10(struct s10 a0) {}

// CHECK: define double @f11()
union { long double a; float b; } f11() { while (1) {} }

// CHECK: define i32 @f12_0()
// CHECK: define void @f12_1(i32 %a0.coerce)
struct s12 { int a __attribute__((aligned(16))); };
struct s12 f12_0(void) { while (1) {} }
void f12_1(struct s12 a0) {}

// Check that sret parameter is accounted for when checking available integer
// registers.
// CHECK: define void @f13(%struct.s13_0* noalias sret %agg.result, i32 %a, i32 %b, i32 %c, i32 %d, {{.*}}* byval align 8 %e, i32 %f)

struct s13_0 { long long f0[3]; };
struct s13_1 { long long f0[2]; };
struct s13_0 f13(int a, int b, int c, int d,
                 struct s13_1 e, int f) { while (1) {} }

// CHECK: define void @f20(%struct.s20* byval align 32 %x)
struct __attribute__((aligned(32))) s20 {
  int x;
  int y;
};
void f20(struct s20 x) {}


// CHECK: declare void @func(i64)
typedef struct _str {
  union {
    long double a;
    long c;
  };
} str;

void func(str s);
str ss;
void f9122143()
{
  func(ss);
}


typedef struct {
  int a;
  int b;
} s1;
// CHECK: define i32 @f48(%struct.s1* byval %s)
int __attribute__((pnaclcall)) f48(s1 s) { return s.a; }

// CHECK: define void @f49(%struct.s1* noalias sret %agg.result)
s1 __attribute__((pnaclcall)) f49() { s1 s; s.a = s.b = 1; return s; }

union simple_union {
  int a;
  char b;
};
// Unions should be passed as byval structs
// CHECK: define void @f50(%union.simple_union* byval %s)
void __attribute__((pnaclcall)) f50(union simple_union s) {}

typedef struct {
  int b4 : 4;
  int b3 : 3;
  int b8 : 8;
} bitfield1;
// Bitfields should be passed as byval structs
// CHECK: define void @f51(%struct.bitfield1* byval %bf1)
void __attribute__((pnaclcall)) f51(bitfield1 bf1) {}
