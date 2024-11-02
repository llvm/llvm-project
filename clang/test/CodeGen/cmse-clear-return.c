// RUN: %clang_cc1 -no-opaque-pointers -triple thumbv8m.main   -O0 -mcmse -S -emit-llvm %s -o - | \
// RUN:    FileCheck %s --check-prefixes=CHECK,CHECK-LE,CHECK-LE-NOPT,CHECK-SOFT
// RUN: %clang_cc1 -no-opaque-pointers -triple thumbebv8m.main -O0 -mcmse -S -emit-llvm %s -o - | \
// RUN:    FileCheck %s --check-prefixes=CHECK,CHECK-BE,CHECK-SOFT
// RUN: %clang_cc1 -no-opaque-pointers -triple thumbv8m.main   -O2 -mcmse -S -emit-llvm %s -o - | \
// RUN:    FileCheck %s --check-prefixes=CHECK,CHECK-LE,CHECK-LE-OPT,CHECK-SOFT
// RUN: %clang_cc1 -no-opaque-pointers -triple thumbebv8m.main -O2 -mcmse -S -emit-llvm %s -o - | \
// RUN:    FileCheck %s --check-prefixes=CHECK,CHECK-BE,CHECK-BE-OPT,CHECK-SOFT
// RUN: %clang_cc1 -no-opaque-pointers -triple thumbv8m.main   -O0 -mcmse -S -emit-llvm %s -o - \
// RUN:            -mfloat-abi hard | \
// RUN:    FileCheck %s --check-prefixes=CHECK,CHECK-LE,CHECK-LE-NOPT,CHECK-HARD


//   :        Memory layout                | Mask
// LE: .......1 ........ ........ ........ | 0x00000001/1
// BE: 1....... ........ ........ ........ | 0x80000000/-2147483648
typedef struct T0 {
  int a : 1, : 31;
} T0;

T0 t0;
__attribute__((cmse_nonsecure_entry)) T0 f0(void) { return t0; }
// CHECK:    define {{.*}} @f0()
// CHECK-LE: %[[R:.*]] = and i32 %{{.*}}, 1
// CHECK-BE: %[[R:.*]] = and i32 %{{.*}}, -2147483648
// CHECK:    ret i32 %[[R]]

// LE: ......1. ........ ........ ........ 0x00000002/2
// BE: .1...... ........ ........ ........ 0x40000000/1073741824
typedef struct T1 {
  int : 1, a : 1, : 30;
} T1;

T1 t1;
__attribute__((cmse_nonsecure_entry)) T1 f1(void) { return t1; }
// CHECK:    define {{.*}} @f1()
// CHECK-LE: %[[R:.*]] = and i32 %{{.*}}, 2
// CHECK-BE: %[[R:.*]] = and i32 %{{.*}}, 1073741824
// CHECK:    ret i32 %[[R]]

// LE: ........ .......1 ........ ........ 0x00000100/256
// BE: ........ 1....... ........ ........ 0x00800000/8388608
typedef struct T2 {
  int : 8, a : 1, : 23;
} T2;

T2 t2;
__attribute__((cmse_nonsecure_entry)) T2 f2(void) { return t2; }
// CHECK:    define {{.*}} @f2()
// CHECK-LE: %[[R:.*]] = and i32 %{{.*}}, 256
// CHECK-BE: %[[R:.*]] = and i32 %{{.*}}, 8388608
// CHECK:    ret i32 %[[R]]

// LE: ........ .....1.. ........ ........ 0x00000400/1024
// BE: ........ ..1..... ........ ........ 0x00200000/2097152
typedef struct T3 {
  int : 10, a : 1;
} T3;

T3 t3;
__attribute__((cmse_nonsecure_entry)) T3 f3(void) { return t3; }
// CHECK:    define {{.*}} @f3()
// CHECK-LE: %[[R:.*]] = and i32 %{{.*}}, 1024
// CHECK-BE: %[[R:.*]] = and i32 %{{.*}}, 2097152
// CHECK:    ret i32 %[[R]]

// LE: 11111111 ........ ........ ........ 0x000000ff/255
// BE: 11111111 ........ ........ ........ 0xff000000/-16777216
typedef struct T4 {
  int a : 8, : 24;
} T4;

T4 t4;
__attribute__((cmse_nonsecure_entry)) T4 f4(void) { return t4; }
// CHECK: define {{.*}} @f4()
// CHECK-LE: %[[R:.*]] = and i32 %{{.*}}, 255
// CHECK-BE: %[[R:.*]] = and i32 %{{.*}}, -16777216
// CHECK: ret i32 %[[R]]

// LE: 1111111. .......1 ........ ........ 0x000001fe/510
// BE: .1111111 1....... ........ ........ 0x7f800000/2139095040
typedef struct T5 {
  int : 1, a : 8, : 23;
} T5;

T5 t5;
__attribute__((cmse_nonsecure_entry)) T5 f5(void) { return t5; }
// CHECK:    define {{.*}} @f5()
// CHECK-LE: %[[R:.*]] = and i32 %{{.*}}, 510
// CHECK-BE: %[[R:.*]] = and i32 %{{.*}}, 2139095040
// CHECK:    ret i32 %[[R]]

// LE: 1111111. 11111111 ........ ........ 0x0000fffe/65534
// BE: .1111111 11111111 ........ ........ 0x7fff0000/2147418112
typedef struct T6 {
  int : 1, a : 15, : 16;
} T6;

T6 t6;
__attribute__((cmse_nonsecure_entry)) T6 f6(void) { return t6; }
// CHECK:    define {{.*}} @f6()
// CHECK-LE: %[[R:.*]] = and i32 %{{.*}}, 65534
// CHECK-BE: %[[R:.*]] = and i32 %{{.*}}, 2147418112
// CHECK:    ret i32 %[[R]]

// LE: 1111111. 11111111 .......1 ........ 0x0001fffe/131070
// BE: .1111111 11111111 1....... ........ 0x7fff8000/2147450880
typedef struct T7 {
  int : 1, a : 16, : 15;
} T7;

T7 t7;
__attribute__((cmse_nonsecure_entry)) T7 f7(void) { return t7; }
// CHECK:    define {{.*}} @f7()
// CHECK-LE: %[[R:.*]] = and i32 %{{.*}}, 131070
// CHECK-BE: %[[R:.*]] = and i32 %{{.*}}, 2147450880
// CHECK:    ret i32 %[[R]]

// LE: 11111111 111111.. 11111111 11111111 0xfffffcff/-769
// BE: 11111111 ..111111 11111111 11111111 0xff3fffff/-12582913
typedef struct T8 {
  struct T80 {
    char a;
    char : 2, b : 6;
  } a;
  short b;
} T8;

T8 t8;
__attribute__((cmse_nonsecure_entry)) T8 f8(void) { return t8; }
// CHECK:    define {{.*}} @f8()
// CHECK-LE: %[[R:.*]] = and i32 %{{.*}}, -769
// CHECK-BE: %[[R:.*]] = and i32 %{{.*}}, -12582913
// CHECK:    ret i32 %[[R]]

// LE: ......11 ..111111 ...11111 ........ 0x001f3f03/2047747
// BE: 11...... 111111.. 11111... ........ 0xc0fcf800/-1057163264
typedef struct T9 {
  struct T90 {
    char a : 2;
    char : 0;
    short b : 6;
  } a;
  int b : 5;
} T9;

T9 t9;
__attribute__((cmse_nonsecure_entry)) T9 f9(void) { return t9; }
// CHECK:    define {{.*}} @f9()
// CHECK-LE: %[[R:.*]] = and i32 %{{.*}}, 2047747
// CHECK-BE: %[[R:.*]] = and i32 %{{.*}}, -1057163264
// CHECK:    ret i32 %[[R]]

T9 f91(void) { return t9; }
// CHECK:  define {{.*}} @f91()
// CHECK: %[[R:.*]] = load i32
// CHECK: ret i32 %[[R]]

// LE: 11111111 ........ 11111111 11111111 0xffff00ff/-65281
// BE: 11111111 ........ 11111111 11111111 0xff00ffff/16711681
typedef struct T10 {
  char a;
  short b;
} T10;

T10 t10;
__attribute__((cmse_nonsecure_entry)) T10 f10(void) { return t10; }
// CHECK: define {{.*}} @f10()
// CHECK-LE: %[[R:.*]] = and i32 %{{.*}}, -65281
// CHECK-BE: %[[R:.*]] = and i32 %{{.*}}, -16711681
// CHECK: ret i32 %[[R]]

// LE: 11111111 11111111 11111111 ........ 0x00ffffff/16777215
// BE: 11111111 11111111 11111111 ........ 0xffffff00/-256
typedef struct T11 {
  short a;
  char b;
} T11;

T11 t11;
__attribute__((cmse_nonsecure_entry)) T11 f11(void) { return t11; }
// CHECK: define {{.*}} @f11()
// CHECK-LE: %[[R:.*]] = and i32 %{{.*}}, 16777215
// CHECK-BE: %[[R:.*]] = and i32 %{{.*}}, -256
// CHECK: ret i32 %[[R]]

// LE: 11111111 11111111 11111111 ........ 0x00ffffff/16777215
// BE: 11111111 11111111 11111111 ........ 0xffffff00/-256
typedef struct T12 {
  char a[3];
} T12;

T12 t12;
__attribute__((cmse_nonsecure_entry)) T12 f12(void) { return t12; }
// CHECK:    define {{.*}} @f12()
// CHECK-LE-OPT:  %[[V0:.*]] = load i24, i24* bitcast (%struct.T12* @t12
// CHECK-LE-OPT:  %[[R:.*]] = zext i24 %[[V0]] to i32
// CHECK-LE-NOPT: %[[R:.*]] = and i32 %{{.*}}, 16777215

// CHECK-BE-OPT:  %[[V0:.*]] = load i24, i24* bitcast (%struct.T12* @t12
// CHECK-BE-OPT:  %[[V1:.*]] = zext i24 %[[V0]] to i32
// CHECK-BE-OPT:  %[[R:.*]] = shl nuw i32 %[[V1]], 8
// CHECK:         ret i32 %[[R]]

// LE: 11111111 11111111 11111111 ........ 0x00ffffff/16777215
// BE: 11111111 11111111 11111111 ........ 0xffffff00/-256
typedef struct __attribute__((packed)) T13 {
  char a;
  short b;
} T13;

T13 t13;
__attribute__((cmse_nonsecure_entry)) T13 f13(void) { return t13; }
// CHECK:         define {{.*}} @f13()
// CHECK-LE-OPT:  %[[V0:.*]] = load i24, i24* bitcast (%struct.T13* @t13
// CHECK-LE-OPT:  %[[R:.*]] = zext i24 %[[V0]] to i32
// CHECK-LE-NOPT: %[[R:.*]] = and i32 %{{.*}}, 16777215

// CHECK-BE-OPT:  %[[V0:.*]] = load i24, i24* bitcast (%struct.T13* @t13
// CHECK-BE-OPT:  %[[V1:.*]] = zext i24 %[[V0]] to i32
// CHECK-BE-OPT:  %[[R:.*]] = shl nuw i32 %[[V1]], 8
// CHECK:         ret i32 %[[R]]

typedef struct __attribute__((packed)) T14 {
  short a;
  short b;
} T14;

T14 t14;
__attribute__((cmse_nonsecure_entry)) T14 f14(void) { return t14; }
// CHECK:         define {{.*}} @f14()
// CHECK:         [[R:%.*]] = load
// CHECK-LE-OPT:  ret i32 [[R]]
// CHECK-LE-NOPT: [[AND:%.+]] = and i32 [[R]], -1
// CHECK-LE-NOPT: ret i32 [[AND]]
// CHECK-BE-OPT:  ret i32 [[R]]

// LE: 1111..11 1111..11 11111111 11111111 0xfffff3f3/-3085
// BE: 11..1111 11..1111 11111111 11111111 0xcfcfffff/-808452097
typedef struct T17 {
  struct T170 {
    char a : 2;
    char   : 2, b : 4;
  } a[2];
  char b[2];
  char c[];
} T17;

T17 t17;
__attribute__((cmse_nonsecure_entry)) T17 f17(void) { return t17; }
// CHECK:    define {{.*}} @f17()
// CHECK-LE: %[[R:.*]] = and i32 {{.*}}, -3085
// CHECK-BE: %[[R:.*]] = and i32 {{.*}}, -808452097
// CHECK: ret i32 %[[R]]

typedef struct T21 {
  float a;
} T21;

T21 t21;
__attribute__((cmse_nonsecure_entry)) T21 f21(void) { return t21; }
// CHECK:      define {{.*}} @f21()
// CHECK-SOFT: ret i32
// CHECK-HARD: ret %struct.T21

__attribute__((cmse_nonsecure_entry)) float f22(void) { return 1.0f; }
// CHECK: define {{.*}} @f22()
// CHECK: ret float
