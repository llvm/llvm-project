// RUN: %clang_cc1 -triple thumbv8m.main   -O0 -mcmse -S -emit-llvm %s -o - | \
// RUN:    FileCheck %s --check-prefixes=CHECK,CHECK-LE,CHECK-SOFTFP
// RUN: %clang_cc1 -triple thumbebv8m.main -O0 -mcmse -S -emit-llvm %s -o - | \
// RUN:    FileCheck %s --check-prefixes=CHECK,CHECK-BE,CHECK-SOFTFP
// RUN: %clang_cc1 -triple thumbv8m.main   -O2 -mcmse -S -emit-llvm %s -o - | \
// RUN:    FileCheck %s --check-prefixes=CHECK,CHECK-LE,CHECK-SOFTFP
// RUN: %clang_cc1 -triple thumbebv8m.main -O2 -mcmse -S -emit-llvm %s -o - | \
// RUN:    FileCheck %s --check-prefixes=CHECK,CHECK-BE,CHECK-SOFTFP
// RUN: %clang_cc1 -triple thumbv8m.main   -O0 -mcmse -mfloat-abi hard  \
// RUN:            -S -emit-llvm %s -o - | \
// RUN:    FileCheck %s --check-prefixes=CHECK,CHECK-LE,CHECK-HARDFP

// We don't really need to repeat *all* the test cases from cmse-clear-return.c
// as it won't increase test coverage.

//   :        Memory layout                | Mask
// LE: .......1 ........ ........ ........ | 0x00000001/1
// BE: 1....... ........ ........ ........ | 0x80000000/-2147483648
typedef struct T0 {
  int a : 1, : 31;
} T0;

void __attribute__((cmse_nonsecure_call)) (*g0)(T0);

T0 t0;
void f0() { g0(t0); }
// CHECK:    define {{.*}} @f0()
// CHECK-LE: %[[V0:.*]] = and i32 {{.*}}, 1
// CHECK-BE: %[[V0:.*]] = and i32 {{.*}}, -2147483648
// CHECK:    %[[V1:.*]] = insertvalue [1 x i32] undef, i32 %[[V0]], 0
// CHECK:    call {{.*}} void %0([1 x i32] %[[V1]])

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
void __attribute__((cmse_nonsecure_call)) (*g8)(T8);
void f8() { g8(t8); }
// CHECK:    define {{.*}} @f8()
// CHECK-LE: %[[V0:.*]] = and i32 {{.*}}, -769
// CHECK-BE: %[[V0:.*]] = and i32 {{.*}}, -12582913
// CHECK:    %[[V1:.*]] = insertvalue [1 x i32] undef, i32 %[[V0]], 0
// CHECK:    call {{.*}} void %0([1 x i32] %[[V1]])

// LE(0): 11111111 ........ 11111111 11111111 0xffff00ff/-65281
// LE(4): ...111.. 11111... 11111111 .....111 0x7fff81c/134215708
// BE(0): 11111111 ........ 11111111 11111111 0xff00ffff/-16711681
// BE(4): ..111... ...11111 11111111 111..... 0x381fffe0/941621216
typedef struct T15 {
  char a;
  short b;
  int : 2, c : 3, : 6, d : 16;
} T15;

T15 t15;

void __attribute__((cmse_nonsecure_call)) (*g15_0)(T15);
void f15_0() {
  g15_0(t15);
}
// CHECK: define {{.*}}@f15_0()
// CHECK: %[[FN:.*]] = load {{.*}} @g15_0
// CHECK-LE:  %cmse.clear = and i32 {{.*}}, -65281
// CHECK-BE:  %cmse.clear = and i32 {{.*}}, -16711681
// CHECK: %[[R0:.*]] = insertvalue [2 x i32] undef, i32 %cmse.clear, 0
// CHECK-LE: %cmse.clear1 = and i32 {{.*}}, 134215708
// CHECK-BE: %cmse.clear1 = and i32 {{.*}}, 941621216
// CHECK: %[[R1:.*]] = insertvalue [2 x i32] %[[R0]], i32 %cmse.clear1, 1
// CHECK: call {{.*}} void %[[FN]]([2 x i32] %[[R1]])

void __attribute__((cmse_nonsecure_call)) (*g15_1)(int, int, int, T15);
void f15_1() {
  g15_1(0, 1, 2, t15);
}
// CHECK: define {{.*}}@f15_1()
// CHECK: %[[FN:.*]] = load {{.*}} @g15_1
// CHECK-LE:  %cmse.clear = and i32 {{.*}}, -65281
// CHECK-BE:  %cmse.clear = and i32 {{.*}}, -16711681
// CHECK: %[[R0:.*]] = insertvalue [2 x i32] undef, i32 %cmse.clear, 0
// CHECK-LE: %cmse.clear1 = and i32 {{.*}}, 134215708
// CHECK-BE: %cmse.clear1 = and i32 {{.*}}, 941621216
// CHECK: %[[R1:.*]] = insertvalue [2 x i32] %[[R0]], i32 %cmse.clear1, 1
// CHECK: call {{.*}} void %[[FN]](i32 noundef 0, i32 noundef 1, i32 noundef 2, [2 x i32] %[[R1]])

// LE: 11111111 ........ 11111111 11111111 1111.... ...11111 ........ .111111.
// LE: 0xff00fffff01f007e/9079291968726434047
// BE: 11111111 ........ 11111111 11111111 ....1111 11111... ........ .111111.
// BE: 0xff00ffff0ff8007e/-71776123088273282

typedef struct T16 {
  char a;
  short b;
  long long : 4, c : 9, : 12, d : 6;
} T16;

T16 t16;

void __attribute__((cmse_nonsecure_call)) (*g16_0)(T16);
void f16_0() {
  g16_0(t16);
}
// CHECK: define {{.*}} @f16_0()
// CHECK: %[[FN:.*]] = load {{.*}} @g16_0
// CHECK-LE: %cmse.clear = and i64 {{.*}}, 9079291968726434047
// CHECK-BE: %cmse.clear = and i64 {{.*}}, -71776123088273282
// CHECK: %[[R:.*]] = insertvalue [1 x i64] undef, i64 %cmse.clear, 0
// CHECK: call {{.*}} void %0([1 x i64] %[[R]])


// LE0: 1111..11 .......1 1111..11 .......1 1111..11 .......1 1111..11 .......1
// LE4: 1111..11 .......1 1111..11 .......1 11111111 11111111 11111111 ........
// LE : 0x01f301f3/32702963 * 3 + 0x00ffffff/16777215
// BE0: 11..1111 1....... 11..1111 1....... 11..1111 1....... 11..1111 1.......
// BE4: 11..1111 1....... 11..1111 1....... 11111111 11111111 11111111 ........
// BE : 0xcf80cf80/-813641856 * 3 + 0xffffff00/-256

typedef struct T18 {
  struct T180 {
    short a : 2;
    short   : 2, b : 5;
  } a[2][3];
  char b[3];
  char c[];
} T18;

T18 t18;

void __attribute__((cmse_nonsecure_call)) (*g18)(T18);
void f18() {
  g18(t18);
}
// CHECK:    define {{.*}} @f18()
// CHECK:    %[[FN:.*]] = load {{.*}} @g18
// CHECK-LE: %cmse.clear = and i32 {{.*}}, 32702963
// CHECK-BE: %cmse.clear = and i32 {{.*}}, -813641856
// CHECK:    %[[R0:.*]] = insertvalue [4 x i32] undef, i32 %cmse.clear, 0
// CHECK-LE: %cmse.clear1 = and i32 {{.*}}, 32702963
// CHECK-BE: %cmse.clear1 = and i32 {{.*}}, -813641856
// CHECK:    %[[R1:.*]] = insertvalue [4 x i32] %[[R0]], i32 %cmse.clear1, 1
// CHECK-LE: %cmse.clear2 = and i32 {{.*}}, 32702963
// CHECK-BE: %cmse.clear2 = and i32 {{.*}}, -813641856
// CHECK:    %[[R2:.*]] = insertvalue [4 x i32] %[[R1]], i32 %cmse.clear2, 2
// CHECK-LE: %cmse.clear3 = and i32 {{.*}}, 16777215
// CHECK-BE: %cmse.clear3 = and i32 {{.*}}, -256
// CHECK:    %[[R3:.*]] = insertvalue [4 x i32] %[[R2]], i32 %cmse.clear3, 3
// CHECK:    call {{.*}} void %[[FN]]([4 x i32] %[[R3]])

// LE: 11111111 11111111 ..111... ..111... 0x3838ffff/943259647
// BE: 11111111 11111111 ...111.. ...111.. 0xffff1c1c/-58340
typedef union T19 {
  short a;
  struct T190 {
    char : 3, a : 3;
  } b[4];
} T19;

T19 t19;
void __attribute__((cmse_nonsecure_call)) (*g19)(T19);
void f19() {
  g19(t19);
}
// CHECK:    define {{.*}} @f19()
// CHECK:    %[[FN:.*]] = load {{.*}} @g19
// CHECK-LE: %cmse.clear = and i32 {{.*}}, 943259647
// CHECK-BE: %cmse.clear = and i32 {{.*}}, -58340
// CHECK:    %[[R:.*]] = insertvalue [1 x i32] undef, i32 %cmse.clear, 0
// CHECK:    call {{.*}} void %[[FN]]([1 x i32] %[[R]])


typedef struct T20 {
  float a[2];
} T20;

T20 t20;
void __attribute__((cmse_nonsecure_call)) (*g20)(T20);
void f20() {
  g20(t20);
}
// CHECK: define {{.*}} @f20()
// CHECK:    %[[FN:.*]] = load {{.*}} @g20
// CHECK-SOFTFP: call arm_aapcscc void %[[FN]]([2 x i32]
// CHECK-HARDFP: call arm_aapcs_vfpcc void %[[FN]](%struct.T20
