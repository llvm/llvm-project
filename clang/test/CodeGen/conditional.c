// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

float test1(int cond, float a, float b) {
  return cond ? a : b;
}

double test2(int cond, float a, double b) {
  return cond ? a : b;
}

void f(void);

void test3(void){
   1 ? f() : (void)0;
}

void test4(void) {
  int i; short j;
  float* k = 1 ? &i : &j;
}

void test5(void) {
  const int* cip;
  void* vp;
  cip = 0 ? vp : cip;
}

void test6(void);
void test7(int);
void* test8(void) {return 1 ? test6 : test7;}


void _efree(void *ptr);
void free(void *ptr);

void _php_stream_free3(void) {
  (1 ? free(0) : _efree(0));
}

void _php_stream_free4(void) {
  1 ? _efree(0) : free(0);
}

// PR5526
struct test9 { int a; };
void* test9spare(void);
void test9(struct test9 *p) {
  p ? p : test9spare();
}

// CHECK: @test10
// CHECK: select i1 {{.*}}, i32 4, i32 5
int test10(int c) {
  return c ? 4 : 5;
}
enum { Gronk = 5 };

// rdar://9289603
// CHECK: @test11
// CHECK: select i1 {{.*}}, i32 4, i32 5
int test11(int c) {
  return c ? 4 : Gronk;
}

// CHECK: @test12
// CHECK: select i1 {{.*}}, double 4.0{{.*}}, double 2.0
double test12(int c) {
  return c ? 4.0 : 2.0;
}
// CHECK: @test13
// CHECK: call {{.*}} @f2(
int f2(void);
void test13(void) {
  f2() ? (void)0 : (void)0;
}
