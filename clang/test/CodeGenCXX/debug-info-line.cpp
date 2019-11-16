// RUN: %clang_cc1 -w -debug-info-kind=line-tables-only -std=c++11 -fexceptions -fcxx-exceptions -S -mllvm -no-discriminators -emit-llvm %s -o - -triple %itanium_abi_triple | FileCheck %s
// RUN: %clang_cc1 -w -debug-info-kind=line-tables-only -std=c++11 -fexceptions -fcxx-exceptions -S -mllvm -no-discriminators -emit-llvm %s -o - -triple i686-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -w -debug-info-kind=line-directives-only -std=c++11 -fexceptions -fcxx-exceptions -S -mllvm -no-discriminators -emit-llvm %s -o - -triple %itanium_abi_triple | FileCheck %s
// RUN: %clang_cc1 -w -debug-info-kind=line-directives-only -std=c++11 -fexceptions -fcxx-exceptions -S -mllvm -no-discriminators -emit-llvm %s -o - -triple i686-linux-gnu | FileCheck %s

int &src();
int *sink();
extern "C" __complex float complex_src();
extern "C" __complex float *complex_sink();

// CHECK-LABEL: define
void f1() {
  *sink()
  // CHECK: store {{.*}}, !dbg [[DBG_F1:!.*]]
#line 100
      = //
      src();
}

struct foo {
  int i;
  int &j;
  __complex float k;
  foo();
};

// CHECK-LABEL: define
foo::foo()
    :
#line 200
      i // CHECK: store i32 {{.*}} !dbg [[DBG_FOO_VALUE:!.*]]
      (src()),
      j // CHECK: store i32* {{.*}} !dbg [[DBG_FOO_REF:!.*]]
      (src()),
      k // CHECK: store float {{.*}} !dbg [[DBG_FOO_COMPLEX:!.*]]
      (complex_src()) {
}

// CHECK-LABEL: define {{.*}}f2{{.*}}
void f2() {
  // CHECK: store float {{.*}} !dbg [[DBG_F2:!.*]]
  *complex_sink()
#line 300
      = //
      complex_src();
}

// CHECK-LABEL: define
void f3() {
  // CHECK: store float {{.*}} !dbg [[DBG_F3:!.*]]
  *complex_sink()
#line 400
      += //
      complex_src();
}

// CHECK-LABEL: define
void f4() {
#line 500
  auto x // CHECK: store {{.*}} !dbg [[DBG_F4:!.*]]
      = src();
}

// CHECK-LABEL: define
void f5() {
#line 600
  auto x // CHECK: store float {{.*}} !dbg [[DBG_F5:!.*]]
      = complex_src();
}

struct agg { int i; };
agg agg_src();

// CHECK-LABEL: define
void f6() {
  agg x;
  // CHECK: call void @llvm.memcpy{{.*}} !dbg [[DBG_F6:!.*]]
  x
#line 700
      = //
      agg_src();
}

// CHECK-LABEL: define
void f7() {
  int *src1();
  int src2();
#line 800
  int x = ( // CHECK: load {{.*}} !dbg [[DBG_F7:!.*]]
      src1())[src2()];
}

// CHECK-LABEL: define
void f8() {
  int src1[1];
  int src2();
#line 900
  int x = ( // CHECK: load {{.*}} !dbg [[DBG_F8:!.*]]
      src1)[src2()];
}

// CHECK-LABEL: define
void f9(int i) {
  int src1[1][i];
  int src2();
#line 1000
  auto x = ( // CHECK: getelementptr {{.*}} !dbg [[DBG_F9:!.*]]
      src1)[src2()];
}

inline void *operator new(decltype(sizeof(1)), void *p) noexcept { return p; }

// CHECK-LABEL: define
void f10() {
  void *void_src();
  (
    // CHECK: store {{.*}} !dbg [[DBG_F10_STORE:!.*]]
#line 1100
      new (void_src()) int(src()));
}

// noexcept just to simplify the codegen a bit
void fn() noexcept(true);

struct bar {
  bar();
  // noexcept(false) to convolute the global dtor
  ~bar() noexcept(false);
};
// global ctor cleanup
// CHECK-LABEL: define
// CHECK: invoke{{ }}
// CHECK: invoke{{ }}
// CHECK:   to label {{.*}}, !dbg [[DBG_GLBL_CTOR_B:!.*]]

// terminate caller
// CHECK-LABEL: define

// global dtor cleanup
// CHECK-LABEL: define
// CHECK: invoke{{ }}
// CHECK: invoke{{ }}
// CHECK:   to label {{.*}}, !dbg [[DBG_GLBL_DTOR_B:!.*]]
#line 1200
bar b[1] = { //
    (fn(),   //
     bar())};

// CHECK-LABEL: define
__complex double f11() {
  __complex double f;
// CHECK: store {{.*}} !dbg [[DBG_F11:!.*]]
#line 1300
  return f;
}

// CHECK-LABEL: define
void f12() {
  int f12_1();
  void f12_2(int = f12_1());
// CHECK: call {{.*}}{{(signext )?}}i32 {{.*}} !dbg [[DBG_F12:!.*]]
#line 1400
  f12_2();
}

// CHECK-LABEL: define
void f13() {
// CHECK: call {{.*}} !dbg [[DBG_F13:!.*]]
#define F13_IMPL 1, src()
  1,
#line 1500
  F13_IMPL;
}

struct f14_impl {
  f14_impl(int);
};

// CHECK-LABEL: define
struct f14_use {
// CHECK: call {{.*}}f14_impl{{.*}}, !dbg [[DBG_F14_CTOR_CALL:![0-9]*]]
#line 1600
  f14_impl v{//
             1};
  f14_use();
};

f14_use::f14_use() = default;

// CHECK-LABEL: define
// CHECK-LABEL: define
void func(foo);
void f15(foo *f) {
  func(
// CHECK: getelementptr {{.*}}, !dbg [[DBG_F15:![0-9]*]]
#line 1700
      f[3]);
}

// CHECK-LABEL: define
void f16(__complex float f) {
  __complex float g = //
// CHECK: add {{.*}}, !dbg [[DBG_F16:![0-9]*]]
#line 1800
      f + 1;
}

// CHECK-LABEL: define
void f17(int *x) {
  1,
// CHECK: getelementptr {{.*}}, !dbg [[DBG_F17:![0-9]*]]
#line 1900
      x[1];
}

// CHECK-LABEL: define
void f18(int a, int b) {
// CHECK: icmp {{.*}}, !dbg [[DBG_F18_1:![0-9]*]]
// CHECK: br {{.*}}, !dbg [[DBG_F18_2:![0-9]*]]
#line 2000
  if (a  //
      && //
      b)
    ;
}

// CHECK-LABEL: define
void f19(int a, int b) {
// CHECK: icmp {{.*}}, !dbg [[DBG_F19_1:![0-9]*]]
// CHECK: br {{.*}}, !dbg [[DBG_F19_2:![0-9]*]]
#line 2100
  if (a  //
      || //
      b)
    ;
}

// CHECK-LABEL: define
void f20(int a, int b, int c) {
// CHECK: icmp {{.*}}, !dbg [[DBG_F20_1:![0-9]*]]
// FIXME: Conditional operator's exprloc should be the '?' not the start of the
// expression, then this would go in the right place. (but adding getExprLoc to
// the ConditionalOperator breaks the ARC migration tool - need to investigate
// further).
// CHECK: br {{.*}}, !dbg [[DBG_F20_1]]
#line 2200
  if (a  //
      ? //
      b : c)
    ;
}

// CHECK-LABEL: define
int f21_a(int = 0);
void f21_b(int = f21_a());
void f21() {
// CHECK: call {{.*}}f21_b{{.*}}, !dbg [[DBG_F21:![0-9]*]]
#line 2300
  f21_b();
}

// CHECK-LABEL: define
struct f22_dtor {
  ~f22_dtor();
};
void f22() {
  {
    f22_dtor f;
    src();
// CHECK: invoke {{.*}}src
// CHECK: call {{.*}}, !dbg [[DBG_F22:![0-9]*]]
// CHECK: call {{.*}}, !dbg [[DBG_F22]]
#line 2400
  }
}

// CHECK-LABEL: define
struct f23_struct {
};
f23_struct f23_a();
void f23_b(f23_struct = f23_a());
void f23() {
// CHECK: call {{.*}}f23_a{{.*}}, !dbg [[DBG_F23:![0-9]*]]
#line 2500
  f23_b();
}

// CHECK-LABEL: define
void f24_a(__complex float = complex_src());
void f24() {
// CHECK: call {{.*}}complex_src{{.*}}, !dbg [[DBG_F24:![0-9]*]]
#line 2600
  f24_a();
}

// CHECK-LABEL: define
void f25_a(int x = __builtin_LINE()) {}
void f25() {
  // CHECK: call void @_Z5f25_ai(i32 {{(signext )?}}2700)
#line 2700
  f25_a();
}
// CHECK: [[DBG_F1]] = !DILocation(line: 100,
// CHECK: [[DBG_FOO_VALUE]] = !DILocation(line: 200,
// CHECK: [[DBG_FOO_REF]] = !DILocation(line: 202,
// CHECK: [[DBG_FOO_COMPLEX]] = !DILocation(line: 204,
// CHECK: [[DBG_F2]] = !DILocation(line: 300,
// CHECK: [[DBG_F3]] = !DILocation(line: 400,
// CHECK: [[DBG_F4]] = !DILocation(line: 500,
// CHECK: [[DBG_F5]] = !DILocation(line: 600,
// CHECK: [[DBG_F6]] = !DILocation(line: 700,
// CHECK: [[DBG_F7]] = !DILocation(line: 800,
// CHECK: [[DBG_F8]] = !DILocation(line: 900,
// CHECK: [[DBG_F9]] = !DILocation(line: 1000,
// CHECK: [[DBG_F10_STORE]] = !DILocation(line: 1100,
// CHECK: [[DBG_GLBL_CTOR_B]] = !DILocation(line: 1200,
// CHECK: [[DBG_GLBL_DTOR_B]] = !DILocation(line: 0,
// CHECK: [[DBG_F11]] = !DILocation(line: 1300,
// CHECK: [[DBG_F12]] = !DILocation(line: 1400,
// CHECK: [[DBG_F13]] = !DILocation(line: 1500,
// CHECK: [[DBG_F14_CTOR_CALL]] = !DILocation(line: 1600,
// CHECK: [[DBG_F15]] = !DILocation(line: 1700,
// CHECK: [[DBG_F16]] = !DILocation(line: 1800,
// CHECK: [[DBG_F17]] = !DILocation(line: 1900,
// CHECK: [[DBG_F18_1]] = !DILocation(line: 2000,
// CHECK: [[DBG_F18_2]] = !DILocation(line: 2001,
// CHECK: [[DBG_F19_1]] = !DILocation(line: 2100,
// CHECK: [[DBG_F19_2]] = !DILocation(line: 2101,
// CHECK: [[DBG_F20_1]] = !DILocation(line: 2200,
// CHECK: [[DBG_F23]] = !DILocation(line: 2500,
// CHECK: [[DBG_F24]] = !DILocation(line: 2600,
