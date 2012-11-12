// RUN: %clang_cc1 -triple i386-pc-linux-gnu -emit-llvm -o - %s | FileCheck -check-prefix=CHECKBASIC %s
// RUN: %clang_cc1 -triple armv7a-eabi -mfloat-abi hard -emit-llvm -o - %s | FileCheck -check-prefix=CHECKCC %s

int g0;
// CHECKBASIC: @g0 = common global i32 0
static int bar1 = 42;
// CHECKBASIC: @bar1 = internal global i32 42

extern int g1;
extern int g1 __attribute((alias("g0")));
// CHECKBASIC: @g1 = alias i32* @g0

void f0(void) { }
extern void f1(void);
extern void f1(void) __attribute((alias("f0")));
// CHECKBASIC: @f1 = alias void ()* @f0
// CHECKBASIC: define void @f0() nounwind {

// Make sure that aliases cause referenced values to be emitted.
// PR3200
static inline int foo1() { return 0; }
// CHECKBASIC: define internal i32 @foo1()
int foo() __attribute__((alias("foo1")));
int bar() __attribute__((alias("bar1")));

extern int test6();
void test7() { test6(); }  // test6 is emitted as extern.

// test6 changes to alias.
int test6() __attribute__((alias("test7")));

static int inner(int a) { return 0; }
static int inner_weak(int a) { return 0; }
extern __typeof(inner) inner_a __attribute__((alias("inner")));
static __typeof(inner_weak) inner_weak_a __attribute__((weakref, alias("inner_weak")));
// CHECKCC: @inner_a = alias i32 (i32)* @inner
// CHECKCC: define internal arm_aapcs_vfpcc i32 @inner(i32 %a) nounwind {

int outer(int a) { return inner(a); }
// CHECKCC: define arm_aapcs_vfpcc i32 @outer(i32 %a) nounwind {
// CHECKCC: call arm_aapcs_vfpcc  i32 @inner(i32 %{{.*}})

int outer_weak(int a) { return inner_weak_a(a); }
// CHECKCC: define arm_aapcs_vfpcc i32 @outer_weak(i32 %a) nounwind {
// CHECKCC: call arm_aapcs_vfpcc  i32 @inner_weak(i32 %{{.*}})
// CHECKCC: define internal arm_aapcs_vfpcc i32 @inner_weak(i32 %a) nounwind {
