// RUN: %clang_cc1 -fsyntax-only -verify -fblocks -Wno-objc-root-class -std=c++14 %s
@protocol NSObject;

void bar(id(^)(void));
void foo(id <NSObject>(^objectCreationBlock)(void)) {
    return bar(objectCreationBlock); // OK
}

void bar2(id(*)(void));
void foo2(id <NSObject>(*objectCreationBlock)(void)) {
    return bar2(objectCreationBlock); // expected-error{{incompatible pointer types passing 'id<NSObject> (*)()' to parameter of type 'id (*)()'}}
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

// Require that the types of block parameters are complete.
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
// __block variable's type when the variable is captured by an escaping block.
namespace N2 {
  template <int n> struct A {
    A() {}
    A(const A &other) {
      int invalid[-n]; // expected-error 2 {{array with a negative size}}
    }
    void m() {}
  };

  typedef void (^BlockFnTy)();
  void func(BlockFnTy);

  void test1() {
    __block A<1> x; // expected-note {{requested here}}
    func(^{ x.m(); });
  }

  template <int n> void test2() {
    __block A<n> x; // expected-note {{requested here}}
    func(^{ x.m(); });
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

namespace DependentTypenameReturnType {
  // Success case
  template <class T> struct S { typedef int type; };
  template <class T> void f() {
    auto b = ^ typename S<T>::type () { return 0; };
    (void)b;
  }
  template void f<int>();

  // Same as 'f' but with no parameter list. The written block signature is
  // then stored as just the return-type loc (there is no FunctionProtoTypeLoc
  // to look through).
  template <class T> void g() {
    auto b = ^ typename S<T>::type { return 0; };
    (void)b;
  }
  template void g<int>();

  // A block with no explicit return type has no return-type source location, so
  // it takes the deduced-return fallback in TransformBlockExpr.
  template <class T> T h() {
    auto b = ^ { T x{}; return x; };
    return b();
  }
  template int h<int>();

  // Failure case
  template <class T> struct NoType {};
  template <class T> void i() {
    auto b = ^ typename NoType<T>::type () { return 0; }; // expected-error{{no type named 'type' in 'DependentTypenameReturnType::NoType<int>'}}
    (void)b;
  }
  template void i<int>(); // expected-note{{in instantiation of function template specialization 'DependentTypenameReturnType::i<int>' requested here}}
}

namespace GenericLambdaCapture {
int test(int outerp) {
  auto lambda =[&](auto p) {
    return ^{
      return p + outerp;
    }();
  };
  return lambda(1);
}
}

namespace MoveBlockVariable {
struct B0 {
};

struct B1 { // expected-note 2 {{candidate constructor (the implicit}}
  B1(B0&&); // expected-note {{candidate constructor not viable}}
};

B1 test_move() {
  __block B0 b;
  return b; // expected-error {{no viable conversion from returned value of type 'B0' to function return type 'B1'}}
}
}
