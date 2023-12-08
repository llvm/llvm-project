// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify %s

// The object declared in an exception-declaration or, if the
// exception-declaration does not specify a name, a temporary (12.2)
// is copy-initialized (8.5) from the exception object.
//
template<typename T>
class X {
  T* ptr;

public:
  X(const X<T> &) {
    int *ip = 0;
    ptr = ip; // expected-error{{incompatible pointer types assigning to 'float *' from 'int *'}}
  }

  ~X() {
    float *fp = 0;
    ptr = fp; // expected-error{{incompatible pointer types assigning to 'int *' from 'float *'}}
  }
};

void f() {
  try {
  } catch (X<float>) { // expected-note{{instantiation}}
    // copy constructor
  } catch (X<int> xi) { // expected-note{{instantiation}}
    // destructor
  }
}

struct Abstract {
  virtual void f() = 0; // expected-note{{pure virtual}}
};

void g() {
  try {
  } catch (Abstract) { // expected-error{{variable type 'Abstract' is an abstract class}}
  }
}
