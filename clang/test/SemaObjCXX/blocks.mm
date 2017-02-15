// RUN: %clang_cc1 -fsyntax-only -verify -fblocks -Wno-objc-root-class -std=c++11 %s
@protocol NSObject;

void bar(id(^)(void));
void foo(id <NSObject>(^objectCreationBlock)(void)) {
    return bar(objectCreationBlock); // OK
}

void bar2(id(*)(void));
void foo2(id <NSObject>(*objectCreationBlock)(void)) {
    return bar2(objectCreationBlock); // expected-warning{{incompatible pointer types passing 'id<NSObject> (*)()' to parameter of type 'id (*)()'}}
}

void bar3(id(*)()); // expected-note{{candidate function}}
void foo3(id (*objectCreationBlock)(int)) {
    return bar3(objectCreationBlock); // expected-error{{no matching}}
}

void bar4(id(^)()); // expected-note{{candidate function}}
void foo4(id (^objectCreationBlock)(int)) {
    return bar4(objectCreationBlock); // expected-error{{no matching}}
}

void foo5(id (^x)(int)) {
  if (x) { }
}

// <rdar://problem/6590445>
@interface Foo {
    @private
    void (^_block)(void);
}
- (void)bar;
@end

namespace N {
  class X { };      
  void foo(X);
}

@implementation Foo
- (void)bar {
    _block();
    foo(N::X()); // okay
}
@end

typedef signed char BOOL;
void foo6(void *block) {  
	void (^vb)(id obj, int idx, BOOL *stop) = (void (^)(id, int, BOOL *))block;
    BOOL (^bb)(id obj, int idx, BOOL *stop) = (BOOL (^)(id, int, BOOL *))block;
}

// <rdar://problem/8600419>: Require that the types of block
// parameters are complete.
namespace N1 {
  template<class _T> class ptr; // expected-note{{template is declared here}}

  template<class _T>
    class foo {
  public:
    void bar(void (^)(ptr<_T>));
  };

  class X;

  void test2();

  void test()
  {
    foo<X> f;
    f.bar(^(ptr<X> _f) { // expected-error{{implicit instantiation of undefined template 'N1::ptr<N1::X>'}}
        test2();
      });
  }
}

// Make sure we successfully instantiate the copy constructor of a
// __block variable's type.
namespace N2 {
  template <int n> struct A {
    A() {}
    A(const A &other) {
      int invalid[-n]; // expected-error 2 {{array with a negative size}}
    }
  };

  void test1() {
    __block A<1> x; // expected-note {{requested here}}
  }

  template <int n> void test2() {
    __block A<n> x; // expected-note {{requested here}}
  }
  template void test2<2>();
}

// Handle value-dependent block declaration references.
namespace N3 {
  template<int N> struct X { };

  template<int N>
  void f() {
    X<N> xN = ^() { return X<N>(); }();
  }
}

// rdar://8979379

@interface A
@end

@interface B : A
@end

void f(int (^bl)(A* a)); // expected-note {{candidate function not viable: no known conversion from 'int (^)(B *)' to 'int (^)(A *)' for 1st argument}}

void g() {
  f(^(B* b) { return 0; }); // expected-error {{no matching function for call to 'f'}}
}

namespace DependentReturn {
  template<typename T>
  void f(T t) {
    (void)^(T u) {
      if (t != u)
        return t + u;
      else
        return;
    };

    (void)^(T u) {
      if (t == u)
        return;
      else
        return t + u;
    };
  }

  struct X { };
  void operator+(X, X);
  bool operator==(X, X);
  bool operator!=(X, X);

  template void f<X>(X);
}

namespace MoveBlockVariable {
struct B0 {
};

struct B1 { // expected-note 2 {{candidate constructor (the implicit}}
  B1(B0&&); // expected-note {{candidate constructor not viable}}
};

B1 test_move() {
  __block B0 b;
  return b; // expected-error {{no viable conversion from returned value of type 'MoveBlockVariable::B0' to function return type 'MoveBlockVariable::B1'}}
}
}
