// RUN: %clang_cc1 -triple wasm32-unknown-unknown %s -emit-llvm -o - \
// RUN:   | FileCheck %s -check-prefix=WEBASSEMBLY32
// RUN: %clang_cc1 -triple wasm64-unknown-unknown %s -emit-llvm -o - \
// RUN:   | FileCheck %s -check-prefix=WEBASSEMBLY64

// Basic argument/attribute tests for WebAssembly

// WEBASSEMBLY32: define void @f0(i32 %i, i32 %j, i64 %k, double %l, fp128 %m)
// WEBASSEMBLY64: define void @f0(i32 %i, i64 %j, i64 %k, double %l, fp128 %m)
void f0(int i, long j, long long k, double l, long double m) {}

typedef struct {
  int aa;
  int bb;
} s1;
// Structs should be passed byval and not split up.
// WEBASSEMBLY32: define void @f1(%struct.s1* byval align 4 %i)
// WEBASSEMBLY64: define void @f1(%struct.s1* byval align 4 %i)
void f1(s1 i) {}

typedef struct {
  int cc;
} s2;
// Single-element structs should be returned as the one element.
// WEBASSEMBLY32: define i32 @f2()
// WEBASSEMBLY64: define i32 @f2()
s2 f2() {
  s2 foo;
  return foo;
}

typedef struct {
  int cc;
  int dd;
} s3;
// Structs should be returned sret and not simplified by the frontend.
// WEBASSEMBLY32: define void @f3(%struct.s3* noalias sret %agg.result)
// WEBASSEMBLY64: define void @f3(%struct.s3* noalias sret %agg.result)
s3 f3() {
  s3 foo;
  return foo;
}

// WEBASSEMBLY32: define void @f4(i64 %i)
// WEBASSEMBLY64: define void @f4(i64 %i)
void f4(long long i) {}

// i8/i16 should be signext, i32 and higher should not.
// WEBASSEMBLY32: define void @f5(i8 signext %a, i16 signext %b)
// WEBASSEMBLY64: define void @f5(i8 signext %a, i16 signext %b)
void f5(char a, short b) {}

// WEBASSEMBLY32: define void @f6(i8 zeroext %a, i16 zeroext %b)
// WEBASSEMBLY64: define void @f6(i8 zeroext %a, i16 zeroext %b)
void f6(unsigned char a, unsigned short b) {}


enum my_enum {
  ENUM1,
  ENUM2,
  ENUM3,
};
// Enums should be treated as the underlying i32.
// WEBASSEMBLY32: define void @f7(i32 %a)
// WEBASSEMBLY64: define void @f7(i32 %a)
void f7(enum my_enum a) {}

enum my_big_enum {
  ENUM4 = 0xFFFFFFFFFFFFFFFF,
};
// Big enums should be treated as the underlying i64.
// WEBASSEMBLY32: define void @f8(i64 %a)
// WEBASSEMBLY64: define void @f8(i64 %a)
void f8(enum my_big_enum a) {}

union simple_union {
  int a;
  char b;
};
// Unions should be passed as byval structs.
// WEBASSEMBLY32: define void @f9(%union.simple_union* byval align 4 %s)
// WEBASSEMBLY64: define void @f9(%union.simple_union* byval align 4 %s)
void f9(union simple_union s) {}

typedef struct {
  int b4 : 4;
  int b3 : 3;
  int b8 : 8;
} bitfield1;
// Bitfields should be passed as byval structs.
// WEBASSEMBLY32: define void @f10(%struct.bitfield1* byval align 4 %bf1)
// WEBASSEMBLY64: define void @f10(%struct.bitfield1* byval align 4 %bf1)
void f10(bitfield1 bf1) {}
