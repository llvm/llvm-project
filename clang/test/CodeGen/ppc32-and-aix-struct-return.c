// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc-unknown-freebsd \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-SVR4
// RUN: %clang_cc1 -triple powerpc-unknown-aix \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-AIX
// RUN: %clang_cc1 -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-AIX
// RUN: %clang_cc1 -triple powerpc-unknown-linux \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-AIX
// RUN: %clang_cc1 -triple powerpc-unknown-linux -maix-struct-return \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-AIX
// RUN: %clang_cc1 -triple powerpc-unknown-linux -msvr4-struct-return \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-SVR4
// RUN: %clang_cc1 -triple powerpc-unknown-netbsd \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-SVR4
// RUN: %clang_cc1 -triple powerpc-unknown-openbsd \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-SVR4
// RUN: %clang_cc1 -triple powerpc-unknown-openbsd -maix-struct-return \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-AIX
// RUN: %clang_cc1 -triple powerpc-unknown-openbsd -msvr4-struct-return \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-SVR4

typedef struct {
} Zero;
typedef struct {
  char c;
} One;
typedef struct {
  short s;
} Two;
typedef struct {
  char c[3];
} Three;
typedef struct {
  float f;
} Four; // svr4 to return i32, not float
typedef struct {
  char c[5];
} Five;
typedef struct {
  short s[3];
} Six;
typedef struct {
  char c[7];
} Seven;
typedef struct {
  int i;
  char c;
} Eight; // padded for alignment
typedef struct {
  char c[9];
} Nine;

// CHECK-AIX-LABEL:  define void @ret0(%struct.Zero* noalias sret(%struct.Zero) {{[^,]*}})
// CHECK-SVR4-LABEL: define void @ret0()
Zero ret0(void) { return (Zero){}; }

// CHECK-AIX-LABEL:  define void @ret1(%struct.One* noalias sret(%struct.One) {{[^,]*}})
// CHECK-SVR4-LABEL: define i8 @ret1()
One ret1(void) { return (One){'a'}; }

// CHECK-AIX-LABEL:  define void @ret2(%struct.Two* noalias sret(%struct.Two) {{[^,]*}})
// CHECK-SVR4-LABEL: define i16 @ret2()
Two ret2(void) { return (Two){123}; }

// CHECK-AIX-LABEL:  define void @ret3(%struct.Three* noalias sret(%struct.Three) {{[^,]*}})
// CHECK-SVR4-LABEL: define i24 @ret3()
Three ret3(void) { return (Three){"abc"}; }

// CHECK-AIX-LABEL:  define void @ret4(%struct.Four* noalias sret(%struct.Four) {{[^,]*}})
// CHECK-SVR4-LABEL: define i32 @ret4()
Four ret4(void) { return (Four){0.4}; }

// CHECK-AIX-LABEL:  define void @ret5(%struct.Five* noalias sret(%struct.Five) {{[^,]*}})
// CHECK-SVR4-LABEL: define i40 @ret5()
Five ret5(void) { return (Five){"abcde"}; }

// CHECK-AIX-LABEL:  define void @ret6(%struct.Six* noalias sret(%struct.Six) {{[^,]*}})
// CHECK-SVR4-LABEL: define i48 @ret6()
Six ret6(void) { return (Six){12, 34, 56}; }

// CHECK-AIX-LABEL:  define void @ret7(%struct.Seven* noalias sret(%struct.Seven) {{[^,]*}})
// CHECK-SVR4-LABEL: define i56 @ret7()
Seven ret7(void) { return (Seven){"abcdefg"}; }

// CHECK-AIX-LABEL:  define void @ret8(%struct.Eight* noalias sret(%struct.Eight) {{[^,]*}})
// CHECK-SVR4-LABEL: define i64 @ret8()
Eight ret8(void) { return (Eight){123, 'a'}; }

// CHECK-AIX-LABEL:  define void @ret9(%struct.Nine* noalias sret(%struct.Nine) {{[^,]*}})
// CHECK-SVR4-LABEL: define void @ret9(%struct.Nine* noalias sret(%struct.Nine) {{[^,]*}})
Nine ret9(void) { return (Nine){"abcdefghi"}; }
