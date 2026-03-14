// RUN: %clang_cc1 -triple i386-unknown-unknown %s -emit-llvm -o - -O3 -no-struct-path-tbaa | FileCheck %s
// RUN: %clang_cc1 -triple i386-unknown-unknown %s -emit-llvm -o - -O3 | FileCheck %s --check-prefix=PATH

static int f0(int n) {
  struct s0 {
    int a : 30;
    int b : 2;
    long long c : 31;
  } x = { 0xdeadbeef, 0xdeadbeef, 0xdeadbeef };
  
  x.a += n;
  x.b += n;
  x.c += n;

  return x.a + x.b + x.c;
}

int g0(void) {
// CHECK-LABEL: @g0()
// CHECK: ret i32 1
// PATH-LABEL: @g0()
// PATH: ret i32 1
  return f0(-1) + 44335655;
}

static int f1(void) {
  struct s1 { 
    int a:13; 
    char b; 
    unsigned short c:7;
  } x;
  
  x.a = -40;
  x.b = 10;
  x.c = 15;

  return x.a + x.b + x.c;
}

int g1(void) {
// CHECK-LABEL: @g1()
// CHECK: ret i32 1
// PATH-LABEL: @g1()
// PATH: ret i32 1
  return f1() + 16;
}

static int f2(void) {
  struct s2 {
    short a[3];
    int b : 15;
  } x;
  
  x.a[0] = x.a[1] = x.a[2] = -40;
  x.b = 10;

  return x.b;
}

int g2(void) {
// CHECK-LABEL: @g2()
// CHECK: ret i32 1
// PATH-LABEL: @g2()
// PATH: ret i32 1
  return f2() - 9;
}

static int f3(int n) {
  struct s3 {
    unsigned a:16;
    unsigned b:28 __attribute__ ((packed));
  } x = { 0xdeadbeef, 0xdeadbeef };
  struct s4 {
    signed a:16;
    signed b:28 __attribute__ ((packed));
  } y;
  y.a = -0x56789abcL;
  y.b = -0x56789abcL;
  return ((y.a += x.a += n) + 
          (y.b += x.b += n));
}

int g3(void) {
// CHECK-LABEL: @g3()
// CHECK: ret i32 1
// PATH-LABEL: @g3()
// PATH: ret i32 1
  return f3(20) + 130725747;
}

static int f4(void) {
  struct s5 {
    int b:1;
  } x;
  x.b = 1;
  return x.b;
}

int g4(void) {
// CHECK-LABEL: @g4()
// CHECK: ret i32 1
// PATH-LABEL: @g4()
// PATH: ret i32 1
  return f4() + 2;
}
