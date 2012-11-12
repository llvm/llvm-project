// RUN: %clang_cc1 -fsyntax-only -verify -fexceptions %s
typedef __SIZE_TYPE__ size_t;

// Operator delete template for placement new with global lookup
template<int I>
struct X0 {
  X0();

  static void* operator new(size_t) {
    return I; // expected-error{{cannot initialize}}
  }

  static void operator delete(void*) {
    int *ip = I; // expected-error{{cannot initialize}}
  }
};

void test_X0() {
  // Using the global operator new suppresses the search for a
  // operator delete in the class.
  ::new X0<2>;

  new X0<3>; // expected-note 2{{instantiation}}
}

// Operator delete template for placement new[] with global lookup
template<int I>
struct X1 {
  X1();

  static void* operator new[](size_t) {
    return I; // expected-error{{cannot initialize}}
  }

  static void operator delete[](void*) {
    int *ip = I; // expected-error{{cannot initialize}}
  }
};

void test_X1() {
  // Using the global operator new suppresses the search for a
  // operator delete in the class.
  ::new X1<2> [17];

  new X1<3> [17]; // expected-note 2{{instantiation}}
}
