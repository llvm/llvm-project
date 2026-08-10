// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -o - %s -fsanitize=signed-integer-overflow,unsigned-integer-overflow | FileCheck %s

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef int int4 __attribute__((ext_vector_type(4)));

enum E1 : int {
  a
};

enum E2 : char {
  b
};

// CHECK-LABEL: define{{.*}} signext i8 @_Z4add1
// CHECK-NOT: sadd.with.overflow
char add1(char c) { return c + c; }

// CHECK-LABEL: define{{.*}} zeroext i8 @_Z4add2
// CHECK-NOT: uadd.with.overflow
uchar add2(uchar uc) { return uc + uc; }

// CHECK-LABEL: define{{.*}} i32 @_Z4add3
// CHECK: sadd.with.overflow
int add3(E1 e) { return e + a; }

// CHECK-LABEL: define{{.*}} signext i8 @_Z4add4
// CHECK-NOT: sadd.with.overflow
char add4(E2 e) { return e + b; }

// CHECK-LABEL: define{{.*}} signext i8 @_Z4sub1
// CHECK-NOT: ssub.with.overflow
char sub1(char c) { return c - c; }

// CHECK-LABEL: define{{.*}} zeroext i8 @_Z4sub2
// CHECK-NOT: usub.with.overflow
uchar sub2(uchar uc) { return uc - uc; }

// CHECK-LABEL: define{{.*}} signext i8 @_Z4sub3
// CHECK-NOT: ssub.with.overflow
char sub3(char c) { return -c; }

// Note: -INT_MIN can overflow.
//
// CHECK-LABEL: define{{.*}} i32 @_Z4sub4
// CHECK: ssub.with.overflow
int sub4(int i) { return -i; }

// CHECK-LABEL: define{{.*}} signext i8 @_Z4mul1
// CHECK-NOT: smul.with.overflow
char mul1(char c) { return c * c; }

// CHECK-LABEL: define{{.*}} zeroext i8 @_Z4mul2
// CHECK-NOT: smul.with.overflow
uchar mul2(uchar uc) { return uc * uc; }

// Note: USHRT_MAX * USHRT_MAX can overflow.
//
// CHECK-LABEL: define{{.*}} zeroext i16 @_Z4mul3
// CHECK: smul.with.overflow
ushort mul3(ushort us) { return us * us; }

// CHECK-LABEL: define{{.*}} i32 @_Z4mul4
// CHECK: smul.with.overflow
int mul4(int i, char c) { return i * c; }

// CHECK-LABEL: define{{.*}} i32 @_Z4mul5
// CHECK: smul.with.overflow
int mul5(int i, char c) { return c * i; }

// CHECK-LABEL: define{{.*}} signext i16 @_Z4mul6
// CHECK-NOT: smul.with.overflow
short mul6(short s) { return s * s; }

// CHECK-LABEL: define{{.*}} signext i8 @_Z4div1
// CHECK-NOT: ubsan_handle_divrem_overflow
char div1(char c) { return c / c; }

// CHECK-LABEL: define{{.*}} zeroext i8 @_Z4div2
// CHECK-NOT: ubsan_handle_divrem_overflow
uchar div2(uchar uc) { return uc / uc; }

// CHECK-LABEL: define{{.*}} signext i8 @_Z4div3
// CHECK-NOT: ubsan_handle_divrem_overflow
char div3(char c, int i) { return c / i; }

// CHECK-LABEL: define{{.*}} signext i8 @_Z4div4
// CHECK: ubsan_handle_divrem_overflow
char div4(int i, char c) { return i / c; }

// Note: INT_MIN / -1 can overflow.
//
// CHECK-LABEL: define{{.*}} signext i8 @_Z4div5
// CHECK: ubsan_handle_divrem_overflow
char div5(int i, char c) { return i / c; }

// CHECK-LABEL: define{{.*}} signext i8 @_Z4rem1
// CHECK-NOT: ubsan_handle_divrem_overflow
char rem1(char c) { return c % c; }

// CHECK-LABEL: define{{.*}} zeroext i8 @_Z4rem2
// CHECK-NOT: ubsan_handle_divrem_overflow
uchar rem2(uchar uc) { return uc % uc; }

// CHECK-LABEL: define{{.*}} signext i8 @_Z4rem3
// CHECK: ubsan_handle_divrem_overflow
char rem3(int i, char c) { return i % c; }

// CHECK-LABEL: define{{.*}} signext i8 @_Z4rem4
// CHECK-NOT: ubsan_handle_divrem_overflow
char rem4(char c, int i) { return c % i; }

// CHECK-LABEL: define{{.*}} signext i8 @_Z4inc1
// CHECK-NOT: sadd.with.overflow
char inc1(char c) { return c++ + (char)0; }

// CHECK-LABEL: define{{.*}} zeroext i8 @_Z4inc2
// CHECK-NOT: uadd.with.overflow
uchar inc2(uchar uc) { return uc++ + (uchar)0; }

// CHECK-LABEL: define{{.*}} void @_Z4inc3
// CHECK-NOT: sadd.with.overflow
void inc3(char c) { c++; }

// CHECK-LABEL: define{{.*}} void @_Z4inc4
// CHECK-NOT: uadd.with.overflow
void inc4(uchar uc) { uc++; }

// CHECK-LABEL: define{{.*}} <4 x i32> @_Z4vremDv4_iS_
// CHECK-NOT: ubsan_handle_divrem_overflow
int4 vrem(int4 a, int4 b) { return a % b; }
