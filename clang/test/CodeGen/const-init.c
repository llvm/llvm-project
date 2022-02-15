// RUN: %clang_cc1 -triple i386-pc-linux-gnu -ffreestanding -Wno-pointer-to-int-cast -verify -emit-llvm -o - %s | FileCheck %s

#include <stdint.h>

// Brace-enclosed string array initializers
char a[] = { "asdf" };
// CHECK: @a ={{.*}} global [5 x i8] c"asdf\00"

char a2[2][5] = { "asdf" };
// CHECK: @a2 ={{.*}} global [2 x [5 x i8]] {{\[}}[5 x i8] c"asdf\00", [5 x i8] zeroinitializer]

// Double-implicit-conversions of array/functions (not legal C, but
// clang accepts it for gcc compat).
intptr_t b = a; // expected-warning {{incompatible pointer to integer conversion}}
int c(void);
void *d = c;
intptr_t e = c; // expected-warning {{incompatible pointer to integer conversion}}

int f, *g = __extension__ &f, *h = (1 != 1) ? &f : &f;

union s2 {
  struct {
    struct { } *f0;
  } f0;
};

int g0 = (int)(&(((union s2 *) 0)->f0.f0) - 0);

// CHECK: @g1x ={{.*}} global { double, double } { double 1.000000e+00{{[0]*}}, double 0.000000e+00{{[0]*}} }
_Complex double g1x = 1.0f;
// CHECK: @g1y ={{.*}} global { double, double } { double 0.000000e+00{{[0]*}}, double 1.000000e+00{{[0]*}} }
_Complex double g1y = 1.0fi;
// CHECK: @g1 ={{.*}} global { i8, i8 } { i8 1, i8 10 }
_Complex char g1 = (char) 1 + (char) 10 * 1i;
// CHECK: @g2 ={{.*}} global { i32, i32 } { i32 1, i32 10 }
_Complex int g2 = 1 + 10i;
// CHECK: @g3 ={{.*}} global { float, float } { float 1.000000e+00{{[0]*}}, float 1.000000e+0{{[0]*}}1 }
_Complex float g3 = 1.0 + 10.0i;
// CHECK: @g4 ={{.*}} global { double, double } { double 1.000000e+00{{[0]*}}, double 1.000000e+0{{[0]*}}1 }
_Complex double g4 = 1.0 + 10.0i;
// CHECK: @g5 ={{.*}} global { i32, i32 } zeroinitializer
_Complex int g5 = (2 + 3i) == (5 + 7i);
// CHECK: @g6 ={{.*}} global { double, double } { double -1.100000e+0{{[0]*}}1, double 2.900000e+0{{[0]*}}1 }
_Complex double g6 = (2.0 + 3.0i) * (5.0 + 7.0i);
// CHECK: @g7 ={{.*}} global i32 1
int g7 = (2 + 3i) * (5 + 7i) == (-11 + 29i);
// CHECK: @g8 ={{.*}} global i32 1
int g8 = (2.0 + 3.0i) * (5.0 + 7.0i) == (-11.0 + 29.0i);
// CHECK: @g9 ={{.*}} global i32 0
int g9 = (2 + 3i) * (5 + 7i) != (-11 + 29i);
// CHECK: @g10 ={{.*}} global i32 0
int g10 = (2.0 + 3.0i) * (5.0 + 7.0i) != (-11.0 + 29.0i);

// PR5108
// CHECK: @gv1 ={{.*}} global %struct.anon <{ i32 0, i8 7 }>, align 1
struct {
  unsigned long a;
  unsigned long b:3;
} __attribute__((__packed__)) gv1  = { .a = 0x0, .b = 7,  };

// PR5118
// CHECK: @gv2 ={{.*}} global %struct.anon.0 <{ i8 1, i8* null }>, align 1 
struct {
  unsigned char a;
  char *b;
} __attribute__((__packed__)) gv2 = { 1, (void*)0 };

// Global references
// CHECK: @g11.l0 = internal global i32 ptrtoint (i32 ()* @g11 to i32)
long g11(void) { 
  static long l0 = (long) g11;
  return l0; 
}

// CHECK: @g12 ={{.*}} global i32 ptrtoint (i8* @g12_tmp to i32)
static char g12_tmp;
long g12 = (long) &g12_tmp;

// CHECK: @g13 ={{.*}} global [1 x %struct.g13_s0] [%struct.g13_s0 { i32 ptrtoint (i8* @g12_tmp to i32) }]
struct g13_s0 {
   long a;
};
struct g13_s0 g13[] = {
   { (long) &g12_tmp }
};

// CHECK: @g14 ={{.*}} global i8* inttoptr (i32 100 to i8*)
void *g14 = (void*) 100;

// CHECK: @g15 ={{.*}} global i32 -1
int g15 = (int) (char) ((void*) 0 + 255);

// CHECK: @g16 ={{.*}} global i64 4294967295
long long g16 = (long long) ((void*) 0xFFFFFFFF);

// CHECK: @g17 ={{.*}} global i32* @g15
int *g17 = (int *) ((long) &g15);

// CHECK: @g18.p = internal global [1 x i32*] [i32* @g19]
void g18(void) {
  extern int g19;
  static int *p[] = { &g19 };
}

// CHECK: @g20.l0 = internal global %struct.g20_s1 { %struct.g20_s0* null, %struct.g20_s0** getelementptr inbounds (%struct.g20_s1, %struct.g20_s1* @g20.l0, i32 0, i32 0) }
struct g20_s0;
struct g20_s1 {
  struct g20_s0 *f0, **f1;
};
void *g20(void) {
  static struct g20_s1 l0 = { ((void*) 0), &l0.f0 };
  return l0.f1;
}

// PR4108
struct g21 {int g21;};
const struct g21 g21 = (struct g21){1};

// PR5474
struct g22 {int x;} __attribute((packed));
struct g23 {char a; short b; char c; struct g22 d;};
struct g23 g24 = {1,2,3,4};

// CHECK: @g25.g26 = internal global i8* getelementptr inbounds ([4 x i8], [4 x i8]* @[[FUNC:.*]], i32 0, i32 0)
// CHECK: @[[FUNC]] = private unnamed_addr constant [4 x i8] c"g25\00"
int g25(void) {
  static const char *g26 = __func__;
  return *g26;
}

// CHECK: @g27.x = internal global i8* bitcast (i8** @g27.x to i8*), align 4
void g27(void) { // PR8073
  static void *x = &x;
}

void g28(void) {
  typedef long long v1i64 __attribute((vector_size(8)));
  typedef short v12i16 __attribute((vector_size(24)));
  typedef long double v2f80 __attribute((vector_size(24)));
  // CHECK: @g28.a = internal global <1 x i64> <i64 10>
  // CHECK: @g28.b = internal global <12 x i16> <i16 0, i16 0, i16 0, i16 -32768, i16 16383, i16 0, i16 0, i16 0, i16 0, i16 -32768, i16 16384, i16 0>
  // CHECK: @g28.c = internal global <2 x x86_fp80> <x86_fp80 0xK3FFF8000000000000000, x86_fp80 0xK40008000000000000000>, align 32
  static v1i64 a = (v1i64)10LL;
  static v12i16 b = (v12i16)(v2f80){1,2};
  static v2f80 c = (v2f80)(v12i16){0,0,0,-32768,16383,0,0,0,0,-32768,16384,0};
}

// PR13643
void g29(void) {
  typedef char DCC_PASSWD[2];
  typedef struct
  {
      DCC_PASSWD passwd;
  } DCC_SRVR_NM;
  // CHECK: @g29.a = internal global %struct.DCC_SRVR_NM { [2 x i8] c"@\00" }, align 1
  // CHECK: @g29.b = internal global [1 x i32] [i32 ptrtoint ([5 x i8]* @.str.1 to i32)], align 4
  // CHECK: @g29.c = internal global [1 x i32] [i32 97], align 4
  static DCC_SRVR_NM a = { {"@"} };
  static int b[1] = { "asdf" }; // expected-warning {{incompatible pointer to integer conversion initializing 'int' with an expression of type 'char[5]'}}
  static int c[1] = { L"a" };
}

// PR21300
void g30(void) {
#pragma pack(1)
  static struct {
    int : 1;
    int x;
  } a = {};
  // CHECK: @g30.a = internal global %struct.anon.1 <{ i8 undef, i32 0 }>, align 1
#pragma pack()
}

void g31(void) {
#pragma pack(4)
  static struct {
    short a;
    long x;
    short z;
  } a = {23122, -12312731, -312};
#pragma pack()
  // CHECK: @g31.a = internal global %struct.anon.2 { i16 23122, i32 -12312731, i16 -312 }, align 4
}
