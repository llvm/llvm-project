// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -Wno-objc-root-class %s

struct X { 
  void f() const;
  ~X();
};

@interface A {
  X x_;
}

- (const X&)x;
- (void)setx:(const X&)other;
@end

@implementation A

- (const X&)x { return x_; }
- (void)setx:(const X&)other { x_ = other; }
- (void)method {
  self.x.f();
}
@end

// rdar://problem/10444030
@interface Test2
- (void) setY: (int) y;
- (int) z;
@end
void test2(Test2 *a) {
  auto y = a.y; // expected-error {{expected getter method not found on object of type 'Test2 *'}}
  auto z = a.z;
}

// rdar://problem/10672108
@interface Test3
- (int) length;
@end
void test3(Test3 *t) {
  char vla[t.length] = {}; // expected-error {{variable-sized object may not be initialized}}
  char *heaparray = new char[t.length];
}

// <rdar://problem/10672501>
namespace std {
  template<typename T> void count();
}

@interface Test4
- (X&) prop;
@end

void test4(Test4 *t) {
  (void)const_cast<const X&>(t.prop);
  (void)dynamic_cast<X&>(t.prop);
  (void)reinterpret_cast<int&>(t.prop);
}

@interface Test5 {
@public
  int count;
}
@property int count;
@end

void test5(Test5* t5) {
  if (t5.count < 2) { }
  if (t5->count < 2) { }
}


@interface Test6
+ (Class)class;
- (Class)class;
@end

void test6(Test6 *t6) {
  Class x = t6.class;
  Class x2 = Test6.class;
}

template<typename T>
void test6_template(T *t6) {
  Class x = t6.class;
}

template void test6_template(Test6*);

// rdar://problem/10965735
struct Test7PointerMaker {
  operator char *() const;
};
@interface Test7
- (char*) implicit_property;
- (char) bad_implicit_property;
- (Test7PointerMaker) implicit_struct_property;
@property int *explicit_property;
@property int bad_explicit_property;
@property Test7PointerMaker explicit_struct_property;
@end
void test7(Test7 *ptr) {
  delete ptr.implicit_property;
  delete ptr.bad_implicit_property; // expected-error {{cannot delete expression of type 'char'}}
  delete ptr.explicit_property;
  delete ptr.bad_explicit_property; // expected-error {{cannot delete expression of type 'int'}}
  delete ptr.implicit_struct_property;
  delete ptr.explicit_struct_property;
}

// Make sure the returned value from property assignment is void,
// because there isn't any other viable way to handle it for
// non-trivial classes.
class NonTrivial1 {
public:
	~NonTrivial1();
};
class NonTrivial2 {
public:
	NonTrivial2();
	NonTrivial2(const NonTrivial2&);
};
@interface TestNonTrivial
@property(assign, nonatomic) NonTrivial1 p1;
@property(assign, nonatomic) NonTrivial2 p2;
@end
TestNonTrivial *TestNonTrivialObj;

extern void* VoidType;
extern decltype(TestNonTrivialObj.p1 = NonTrivial1())* VoidType;
extern decltype(TestNonTrivialObj.p2 = NonTrivial2())* VoidType;

