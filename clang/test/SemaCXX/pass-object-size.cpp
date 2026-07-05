// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

namespace simple {
int Foo(void *const p __attribute__((pass_object_size(0))));

int OvlFoo(void *const p __attribute__((pass_object_size(0))));
int OvlFoo(void *const p, int);

struct Statics {
  static int Foo(void *const p __attribute__((pass_object_size(0))));
  static int OvlFoo(void *const p __attribute__((pass_object_size(0))));
  static int OvlFoo(void *const p __attribute__((pass_object_size(1)))); // expected-error{{conflicting pass_object_size attributes on parameters}} expected-note@-1{{previous declaration is here}}
  static int OvlFoo(double *p);
};

struct Members {
  int Foo(void *const p __attribute__((pass_object_size(0))));
  int OvlFoo(void *const p __attribute__((pass_object_size(0))));
  int OvlFoo(void *const p, int);
};

void Decls() {
  int (*A)(void *) = &Foo; //expected-error{{cannot take address of function 'Foo' because parameter 1 has pass_object_size attribute}}
  int (*B)(void *) = Foo; //expected-error{{cannot take address of function 'Foo' because parameter 1 has pass_object_size attribute}}

  int (*C)(void *) = &OvlFoo; //expected-error{{address of overloaded function 'OvlFoo' does not match required type 'int (void *)'}} expected-note@6{{candidate address cannot be taken because parameter 1 has pass_object_size attribute}} expected-note@7{{candidate function has different number of parameters (expected 1 but has 2)}}
  int (*D)(void *) = OvlFoo; //expected-error{{address of overloaded function 'OvlFoo' does not match required type 'int (void *)'}} expected-note@6{{candidate address cannot be taken because parameter 1 has pass_object_size attribute}} expected-note@7{{candidate function has different number of parameters (expected 1 but has 2)}}

  int (*E)(void *) = &Statics::Foo; //expected-error{{cannot take address of function 'Foo' because parameter 1 has pass_object_size attribute}}
  int (*F)(void *) = &Statics::OvlFoo; //expected-error{{address of overloaded function 'OvlFoo' does not match required type 'int (void *)'}} expected-note@11{{candidate address cannot be taken because parameter 1 has pass_object_size attribute}} expected-note@13{{candidate function has type mismatch at 1st parameter (expected 'void *' but has 'double *')}}

  int (*G)(void *) = &Members::Foo; //expected-error{{cannot take address of function 'Foo' because parameter 1 has pass_object_size attribute}}
  int (*H)(void *) = &Members::OvlFoo; //expected-error{{address of overloaded function 'OvlFoo' does not match required type 'int (void *)'}} expected-note@18{{candidate address cannot be taken because parameter 1 has pass_object_size attribute}} expected-note@19{{candidate function has different number of parameters (expected 1 but has 2)}}
}

void Assigns() {
  int (*A)(void *);
  A = &Foo; //expected-error{{cannot take address of function 'Foo' because parameter 1 has pass_object_size attribute}}
  A = Foo; //expected-error{{cannot take address of function 'Foo' because parameter 1 has pass_object_size attribute}}

  A = &OvlFoo; //expected-error{{assigning to 'int (*)(void *)' from incompatible type '<overloaded function type>'}} expected-note@6{{candidate address cannot be taken because parameter 1 has pass_object_size attribute}} expected-note@7{{candidate function has different number of parameters (expected 1 but has 2)}}
  A = OvlFoo; //expected-error{{assigning to 'int (*)(void *)' from incompatible type '<overloaded function type>'}} expected-note@6{{candidate address cannot be taken because parameter 1 has pass_object_size attribute}} expected-note@7{{candidate function has different number of parameters (expected 1 but has 2)}}

  A = &Statics::Foo; //expected-error{{cannot take address of function 'Foo' because parameter 1 has pass_object_size attribute}}
  A = &Statics::OvlFoo; //expected-error{{assigning to 'int (*)(void *)' from incompatible type '<overloaded function type>'}} expected-note@11{{candidate address cannot be taken because parameter 1 has pass_object_size attribute}} expected-note@13{{candidate function has type mismatch at 1st parameter (expected 'void *' but has 'double *')}}

  int (Members::*M)(void *);
  M = &Members::Foo; //expected-error{{cannot take address of function 'Foo' because parameter 1 has pass_object_size attribute}}
  M = &Members::OvlFoo; //expected-error-re{{assigning to '{{.*}}' from incompatible type '<overloaded function type>'}} expected-note@18{{candidate address cannot be taken because parameter 1 has pass_object_size attribute}} expected-note@19{{candidate function has different number of parameters (expected 1 but has 2)}}
}

} // namespace simple

namespace templates {
template <typename T>
int Foo(void *const __attribute__((pass_object_size(0)))) {
  return 0;
}

template <typename T> struct Bar {
  template <typename U>
  int Foo(void *const __attribute__((pass_object_size(0)))) {
    return 0;
  }
};

void Decls() {
  int (*A)(void *) = &Foo<void*>; //expected-error{{address of overloaded function 'Foo' does not match required type 'int (void *)'}} expected-note@56{{candidate address cannot be taken because parameter 1 has pass_object_size attribute}}
  int (Bar<int>::*B)(void *) = &Bar<int>::Foo<double>; //expected-error{{address of overloaded function 'Foo' does not match required type}} expected-note@62{{candidate address cannot be taken because parameter 1 has pass_object_size attribute}}
}

void Assigns() {
  int (*A)(void *);
  A = &Foo<void*>; // expected-error{{assigning to 'int (*)(void *)' from incompatible type '<overloaded function type>'}} expected-note@56{{candidate address cannot be taken because parameter 1 has pass_object_size attribute}}
  int (Bar<int>::*B)(void *) = &Bar<int>::Foo<double>; //expected-error{{address of overloaded function 'Foo' does not match required type}} expected-note@62{{candidate address cannot be taken because parameter 1 has pass_object_size attribute}}
}
} // namespace templates

namespace virt {
struct Foo {
  virtual void DoIt(void *const p __attribute__((pass_object_size(0))));
};

struct Bar : public Foo {
  void DoIt(void *const p __attribute__((pass_object_size(0)))) override; // OK
};

struct Baz : public Foo {
  void DoIt(void *const p) override; //expected-error{{non-virtual member function marked 'override' hides virtual member function}} expected-note@81{{hidden overloaded virtual function 'virt::Foo::DoIt' declared here}}
};
}

namespace why {
void TakeFn(void (*)(int, void *));
void ObjSize(int, void *const __attribute__((pass_object_size(0))));

void Check() {
  TakeFn(ObjSize); //expected-error{{cannot take address of function 'ObjSize' because parameter 2 has pass_object_size attribute}}
  TakeFn(&ObjSize); //expected-error{{cannot take address of function 'ObjSize' because parameter 2 has pass_object_size attribute}}
  TakeFn(*ObjSize); //expected-error{{cannot take address of function 'ObjSize' because parameter 2 has pass_object_size attribute}}
  TakeFn(*****ObjSize); //expected-error{{cannot take address of function 'ObjSize' because parameter 2 has pass_object_size attribute}}
  TakeFn(*****&ObjSize); //expected-error{{cannot take address of function 'ObjSize' because parameter 2 has pass_object_size attribute}}

  void (*P)(int, void *) = ****ObjSize; //expected-error{{cannot take address of function 'ObjSize' because parameter 2 has pass_object_size attribute}}
  P = ****ObjSize; //expected-error{{cannot take address of function 'ObjSize' because parameter 2 has pass_object_size attribute}}

  TakeFn((ObjSize)); //expected-error{{cannot take address of function 'ObjSize' because parameter 2 has pass_object_size attribute}}
  TakeFn((void*)ObjSize); //expected-error{{cannot take address of function 'ObjSize' because parameter 2 has pass_object_size attribute}}
  TakeFn((decltype(P))((void*)ObjSize)); //expected-error{{cannot take address of function 'ObjSize' because parameter 2 has pass_object_size attribute}}
}
}

namespace constexpr_support {
constexpr int getObjSizeType() { return 0; }
void Foo(void *p __attribute__((pass_object_size(getObjSizeType()))));
}

namespace lambdas {
void Bar() {
  (void)+[](void *const p __attribute__((pass_object_size(0)))) {}; //expected-error-re{{invalid argument type '(lambda at {{.*}})' to unary expression}}
}
}

namespace ovlbug {
// Directly calling an address-of function expression (e.g. in (&foo)(args...))
// doesn't go through regular address-of-overload logic. This caused the above
// code to generate an ICE.
void DirectAddrOf(void *__attribute__((pass_object_size(0))));
void DirectAddrOfOvl(void *__attribute__((pass_object_size(0))));
void DirectAddrOfOvl(int *);

void Test() {
  (&DirectAddrOf)(nullptr); //expected-error{{cannot take address of function 'DirectAddrOf' because parameter 1 has pass_object_size attribute}}
  (&DirectAddrOfOvl)((char*)nullptr); //expected-error{{no matching function}} expected-note@129{{candidate address cannot be taken because parameter 1 has pass_object_size attribute}} expected-note@130{{candidate function not viable: no known conversion from 'char *' to 'int *' for 1st argument}}
}
}
