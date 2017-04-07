// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
@interface Foo
@end

@implementation Foo

void func(id);

+ zone {
 func(self);
 return self;
}
@end

@protocol P0
@end

@protocol P1
@end

@interface A <P0>
@end

@interface B : A
@end

@interface C <P1>
@end

int& f(A*); // expected-note {{candidate}}
float& f(B*); // expected-note {{candidate}}
void g(A*);

int& h(A*);
float& h(id);

void test0(A* a, B* b, id val) {
  int& i1 = f(a);
  float& f1 = f(b);

  // GCC succeeds here, which is clearly ridiculous.
  float& f2 = f(val); // expected-error {{ambiguous}}

  g(a);
  g(b);
  g(val);
  int& i2 = h(a);
  float& f3 = h(val);

  int& i3 = h(b);
}

void test1(A* a) {
  B* b = a; // expected-warning{{incompatible pointer types initializing 'B *' with an expression of type 'A *'}}
  B *c; c = a; // expected-warning{{incompatible pointer types assigning to 'B *' from 'A *'}}
}

void test2(A** ap) {
  B** bp = ap; // expected-warning{{incompatible pointer types initializing 'B **' with an expression of type 'A **'}}
  bp = ap; // expected-warning{{incompatible pointer types assigning to 'B **' from 'A **'}}
}

int& cv(A*);
float& cv(const A*);

int& cv2(void*);
float& cv2(const void*);

void cv_test(A* a, B* b, const A* ac, const B* bc) {
  int &i1 = cv(a);
  int &i2 = cv(b);
  float &f1 = cv(ac);
  float &f2 = cv(bc);
  int& i3 = cv2(a);
  float& f3 = cv2(ac);
}

int& qualid(id<P0>);
float& qualid(id<P1>);

void qualid_test(A *a, B *b, C *c) {
  int& i1 = qualid(a);
  int& i2 = qualid(b);

  float& f1 = qualid(c);

  id<P0> p1 = 0;
  p1 = 0;
}


@class NSException;
typedef struct {
    void (*throw_exc)(id);
}
objc_exception_functions_t;

void (*_NSExceptionRaiser(void))(NSException *) {
    objc_exception_functions_t exc_funcs;
    return exc_funcs.throw_exc; // expected-warning{{incompatible pointer types returning 'void (*)(id)' from a function with result type 'void (*)(NSException *)'}}
}

namespace test5 {
  void foo(bool);
  void foo(void *);

  void test(id p) {
    foo(p);
  }
}

// rdar://problem/8592139
namespace test6 {
  void foo(id); // expected-note{{candidate function}}
  void foo(A*) __attribute__((unavailable)); // expected-note {{explicitly made unavailable}}

  void test(B *b) {
    foo(b); // expected-error {{call to unavailable function 'foo'}}
  }
}

namespace rdar8714395 {
  int &f(const void*);
  float &f(const Foo*);

  int &f2(const void*);
  float &f2(Foo const* const *);

  int &f3(const void*);
  float &f3(Foo const**);

  void g(Foo *p) {
    float &fr = f(p);
    float &fr2 = f2(&p);
    int &ir = f3(&p);
  }

  
}

namespace rdar8734046 {
  void f1(id);
  void f2(id<P0>);
  void g(const A *a) {
    f1(a);
    f2(a);
  }
}

namespace PR9735 {
  int &f3(const A*);
  float &f3(const void*);

  void test_f(B* b, const B* bc) {
    int &ir1 = f3(b);
    int &ir2 = f3(bc);
  }
}

@interface D : B
@end

namespace rdar9327203 {
  int &f(void* const&, int);
  float &f(void* const&, long);
  
  void g(id x) { 
    int &fr = (f)(x, 0); 
  }
}

namespace class_id {
  // it's okay to overload Class with id.
  void f(Class) { }
  void f(id) { }
}

@interface NSDictionary<__covariant KeyType, __covariant ObjectType> : A
@end

@interface NSMutableDictionary<KeyType, ObjectType> : NSDictionary<KeyType, ObjectType>
@end

namespace rdar20124827 {

int overload(NSDictionary *) { return 1; }

__attribute__((deprecated))  // expected-note {{'overload' has been explicitly marked deprecated here}}
int overload(NSMutableDictionary *) { return 0; }

__attribute__((deprecated))
void overload2(NSDictionary *); // expected-note {{candidate function}}
void overload2(NSDictionary<A *, A *> *); // expected-note {{candidate function}}

void test(NSDictionary *d1, NSDictionary<A *, A *> *d2, NSMutableDictionary<A *, A *> *m1) {
  overload(d1);
  overload(d2); // no warning
  overload(m1); // expected-warning {{'overload' is deprecated}}
  overload2(d2); // no warning
  overload2(m1); // expected-error {{call to 'overload2' is ambiguous}}
}

}
