// RUN: %clang_cc1 -fsyntax-only -verify -fexceptions %s
typedef __SIZE_TYPE__ size_t;

// Overloaded operator delete with two arguments
template<int I>
struct X0 {
  X0();
  static void* operator new(size_t);
  static void operator delete(void*, size_t) {
    int *ip = I; // expected-error{{cannot initialize}}
  }
};

void test_X0() {
  new X0<1>; // expected-note{{instantiation}}
}

// Overloaded operator delete with one argument
template<int I>
struct X1 {
  X1();

  static void* operator new(size_t);
  static void operator delete(void*) {
    int *ip = I; // expected-error{{cannot initialize}}
  }
};

void test_X1() {
  new X1<1>; // expected-note{{instantiation}}
}

// Overloaded operator delete for placement new
template<int I>
struct X2 {
  X2();

  static void* operator new(size_t, double, double);
  static void* operator new(size_t, int, int);

  static void operator delete(void*, const int, int) {
    int *ip = I; // expected-error{{cannot initialize}}
  }

  static void operator delete(void*, double, double);
};

void test_X2() {
  new (0, 0) X2<1>; // expected-note{{instantiation}}
}

// Operator delete template for placement new
struct X3 {
  X3();

  static void* operator new(size_t, double, double);

  template<typename T>
  static void operator delete(void*, T x, T) {
    double *dp = &x;
    int *ip = &x; // expected-error{{cannot initialize}}
  }
};

void test_X3() {
  new (0, 0) X3; // expected-note{{instantiation}}
}

// Operator delete template for placement new in global scope.
struct X4 {
  X4();
  static void* operator new(size_t, double, double);
};

template<typename T>
void operator delete(void*, T x, T) {
  double *dp = &x;
  int *ip = &x; // expected-error{{cannot initialize}}
}

void test_X4() {
  new (0, 0) X4; // expected-note{{instantiation}}
}

// Useless operator delete hides global operator delete template.
struct X5 {
  X5();
  static void* operator new(size_t, double, double);
  void operator delete(void*, double*, double*);
};

void test_X5() {
  new (0, 0) X5; // okay, we found X5::operator delete but didn't pick it
}

// Operator delete template for placement new
template<int I>
struct X6 {
  X6();

  static void* operator new(size_t) {
    return I; // expected-error{{cannot initialize}}
  }

  static void operator delete(void*) {
    int *ip = I; // expected-error{{cannot initialize}}
  }
};

void test_X6() {
  new X6<3>; // expected-note 2{{instantiation}}
}

void *operator new(size_t, double, double, double);

template<typename T>
void operator delete(void*, T x, T, T) {
  double *dp = &x;
  int *ip = &x; // expected-error{{cannot initialize}}
}
void test_int_new() {
  new (1.0, 1.0, 1.0) int; // expected-note{{instantiation}}
}

// We don't need an operator delete if the type has a trivial
// constructor, since we know that constructor cannot throw.
// FIXME: Is this within the standard? Seems fishy, but both EDG+GCC do it.
#if 0
template<int I>
struct X7 {
  static void* operator new(size_t);
  static void operator delete(void*, size_t) {
    int *ip = I; // okay, since it isn't instantiated.
  }
};

void test_X7() {
  new X7<1>;
}
#endif

