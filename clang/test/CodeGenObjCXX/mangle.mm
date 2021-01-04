// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -std=c++11 -emit-llvm -fblocks -o - | FileCheck %s

// CHECK: @"_ZZ11+[A shared]E1a" = internal global
// CHECK: @"_ZZ11-[A(Foo) f]E1a" = internal global
// CHECK: v56@0:8i16i20i24i28i32i36i40i44^i48

@interface A
@end

@implementation A

+ (A *)shared {
  static A* a;
  
  return a;
}

@end

@interface A(Foo)
@end

@implementation A(Foo)
- (int)f {
  // FIXME: Add a member function to s and make sure that it's mangled correctly.
  struct s {
  };
  
  static s a;

  return 0;
}
@end

// PR6468
@interface Test
- (void) process: (int)r3 :(int)r4 :(int)r5 :(int)r6 :(int)r7 :(int)r8 :(int)r9 :(int)r10 :(int &)i;
@end

@implementation Test
- (void) process: (int)r3 :(int)r4 :(int)r5 :(int)r6 :(int)r7 :(int)r8 :(int)r9 :(int)r10 :(int &)i {
}
@end

// rdar://9566314
@interface NX
- (void)Meth;
@end

@implementation NX
- (void)Meth {
  void uiIsVisible();
// CHECK: call void @_Z11uiIsVisiblev
  uiIsVisible();
}
@end

// rdar://13434937
//
// Don't crash when mangling an enum whose semantic context
// is a class extension (which looks anonymous in the AST).
// The other tests here are just for coverage.
@interface Test2 @end
@interface Test2 ()
@property (assign) enum { T2x, T2y, T2z } axis;
@end
@interface Test2 (a)
@property (assign) enum { T2i, T2j, T2k } dimension;
@end
@implementation Test2 {
@public
  enum { T2a, T2b, T2c } alt_axis;
}
@end
template <class T> struct Test2Template { Test2Template() {} }; // must have a member that we'll instantiate and mangle
void test2(Test2 *t) {
  Test2Template<decltype(t.axis)> t0;
  Test2Template<decltype(t.dimension)> t1;
  Test2Template<decltype(t->alt_axis)> t2;
}

@protocol P;
void overload1(A<P>*) {}
// CHECK-LABEL: define{{.*}} void @_Z9overload1PU11objcproto1P1A
void overload1(const A<P>*) {}
// CHECK-LABEL: define{{.*}} void @_Z9overload1PKU11objcproto1P1A
void overload1(A<P>**) {}
// CHECK-LABEL: define{{.*}} void @_Z9overload1PPU11objcproto1P1A
void overload1(A<P>*const*) {}
// CHECK-LABEL: define{{.*}} void @_Z9overload1PKPU11objcproto1P1A
void overload1(A<P>***) {}
// CHECK-LABEL: define{{.*}} void @_Z9overload1PPPU11objcproto1P1A
void overload1(void (f)(A<P>*)) {}
// CHECK-LABEL: define{{.*}} void @_Z9overload1PFvPU11objcproto1P1AE

template<typename T> struct X { void f(); };
template<> void X<A*>::f() {}
// CHECK-LABEL: define{{.*}} void @_ZN1XIP1AE1fEv
template<> void X<A<P>*>::f() {}
// CHECK-LABEL: define{{.*}} void @_ZN1XIPU11objcproto1P1AE1fEv

// CHECK-LABEL: define{{.*}} void @_Z12kindof_test2PU8__kindof5Test2
void kindof_test2(__kindof Test2 *t2) { }

@interface Parameterized<T, U> : A
@end

// CHECK-LABEL: define{{.*}} void @_Z19parameterized_test1P13ParameterizedIP1AP4TestE
void parameterized_test1(Parameterized<A *, Test *> *p) {}

// CHECK-LABEL: define{{.*}} void @_Z19parameterized_test2PU8__kindof13ParameterizedIP1AP4TestE
void parameterized_test2(__kindof Parameterized<A *, Test *> *p) {}

// CHECK-LABEL: define{{.*}} void @_Z19parameterized_test3P13Parameterized
void parameterized_test3(Parameterized *p) {}

// CHECK-LABEL: define {{.*}}void @_Z1fP11objc_object
void f(__attribute__((ns_consumed)) id) {}
// CHECK-LABEL: define {{.*}}void @_Z1fPFP11objc_objectS0_S0_E
void f(id (*fn)(__attribute__((ns_consumed)) id, id)) {}
// CHECK-LABEL: define {{.*}}void @_Z1fU13block_pointerFvP11objc_objectE
void f(void (^)(__attribute__((ns_consumed)) id)) {}
