// RUN: %clang_cc1 -std=c++98 -emit-llvm %s -o - -triple=x86_64-apple-darwin10 | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=x86_64-apple-darwin10 | FileCheck %s

struct S {
  virtual ~S() { }
};

// PR5706
// Make sure this doesn't crash; the mangling doesn't matter because the name
// doesn't have linkage.
static struct : S { } obj8;

void f() {
  // Make sure this doesn't crash; the mangling doesn't matter because the
  // generated vtable/etc. aren't modifiable (although it would be nice for
  // codesize to make it consistent inside inline functions).
  static struct : S { } obj8;
}

inline int f2() {
  // FIXME: We don't mangle the names of a or x correctly!
  static struct { int a() { static int x; return ++x; } } obj;
  return obj.a();
}

int f3() { return f2(); }

struct A {
  typedef struct { int x; } *ptr;
  ptr m;
  int a() {
    static struct x {
      // FIXME: We don't mangle the names of a or x correctly!
      int a(ptr A::*memp) { static int x; return ++x; }
    } a;
    return a.a(&A::m);
  }
};

int f4() { return A().a(); }

int f5() {
  static union {
    int a;
  };
  
  // CHECK: _ZZ2f5vE1a
  return a;
}

#if __cplusplus <= 199711L
int f6() {
  static union {
    union {
      int : 1;
    };
    int b;
  };
  
  // CXX98: _ZZ2f6vE1b
  return b;
}
#endif

int f7() {
  static union {
    union {
      int b;
    } a;
  };
  
  // CHECK: _ZZ2f7vE1a
  return a.b;
}

// This used to cause an assert because the typedef-for-anonymous-tag
// code was trying to claim the enum for the template.
enum { T8 };
template <class T> struct Test8 {
  typedef T type;
  Test8(type t) {} // tested later
};
template <class T> void make_test8(T value) { Test8<T> t(value); }
void test8() { make_test8(T8); }

// CHECK-LABEL: define internal void @"_ZNV3$_35test9Ev"(
typedef volatile struct {
  void test9() volatile {}
} Test9;
void test9() {
  Test9 a;
  a.test9();
}

// CHECK-LABEL: define internal void @"_ZN5Test8I3$_2EC1ES0_"(
