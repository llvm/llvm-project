// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++20 -verify %s
// RUN: %clang_cc1 -verify=ref %s
// RUN: %clang_cc1 -verify=ref -std=c++20 %s

constexpr int m = 3;
constexpr const int *foo[][5] = {
  {nullptr, &m, nullptr, nullptr, nullptr},
  {nullptr, nullptr, &m, nullptr, nullptr},
  {nullptr, nullptr, nullptr, &m, nullptr},
};

static_assert(foo[0][0] == nullptr, "");
static_assert(foo[0][1] == &m, "");
static_assert(foo[0][2] == nullptr, "");
static_assert(foo[0][3] == nullptr, "");
static_assert(foo[0][4] == nullptr, "");
static_assert(foo[1][0] == nullptr, "");
static_assert(foo[1][1] == nullptr, "");
static_assert(foo[1][2] == &m, "");
static_assert(foo[1][3] == nullptr, "");
static_assert(foo[1][4] == nullptr, "");
static_assert(foo[2][0] == nullptr, "");
static_assert(foo[2][1] == nullptr, "");
static_assert(foo[2][2] == nullptr, "");
static_assert(foo[2][3] == &m, "");
static_assert(foo[2][4] == nullptr, "");


constexpr int SomeInt[] = {1};
constexpr int getSomeInt() { return *SomeInt; }
static_assert(getSomeInt() == 1, "");

/// A init list for a primitive value.
constexpr int f{5};
static_assert(f == 5, "");


constexpr int getElement(int i) {
  int values[] = {1, 4, 9, 16, 25, 36};
  return values[i];
}
static_assert(getElement(1) == 4, "");
static_assert(getElement(5) == 36, "");

constexpr int data[] = {5, 4, 3, 2, 1};
constexpr int getElement(const int *Arr, int index) {
  return *(Arr + index);
}

constexpr int derefPtr(const int *d) {
  return *d;
}
static_assert(derefPtr(data) == 5, "");

constexpr int storePtr() {
  int b[] = {1,2,3,4};
  int *c = b;

  *c = 4;
  return *c;
}
static_assert(storePtr() == 4, "");


static_assert(getElement(data, 1) == 4, "");
static_assert(getElement(data, 4) == 1, "");

constexpr int getElementFromEnd(const int *Arr, int size, int index) {
  return *(Arr + size - index - 1);
}
static_assert(getElementFromEnd(data, 5, 0) == 1, "");
static_assert(getElementFromEnd(data, 5, 4) == 5, "");


constexpr static int arr[2] = {1,2};
constexpr static int arr2[2] = {3,4};
constexpr int *p1 = nullptr;
constexpr int *p2 = p1 + 1; // expected-error {{must be initialized by a constant expression}} \
                            // expected-note {{cannot perform pointer arithmetic on null pointer}} \
                            // ref-error {{must be initialized by a constant expression}} \
                            // ref-note {{cannot perform pointer arithmetic on null pointer}}
constexpr int *p3 = p1 + 0;
constexpr int *p4 = p1 - 0;
constexpr int *p5 =  0 + p1;
constexpr int *p6 =  0 - p1; // expected-error {{invalid operands to binary expression}} \
                             // ref-error {{invalid operands to binary expression}}

constexpr int const * ap1 = &arr[0];
constexpr int const * ap2 = ap1 + 3; // expected-error {{must be initialized by a constant expression}} \
                                     // expected-note {{cannot refer to element 3 of array of 2}} \
                                     // ref-error {{must be initialized by a constant expression}} \
                                     // ref-note {{cannot refer to element 3 of array of 2}}

constexpr auto ap3 = arr - 1; // expected-error {{must be initialized by a constant expression}} \
                              // expected-note {{cannot refer to element -1}} \
                              // ref-error {{must be initialized by a constant expression}} \
                              // ref-note {{cannot refer to element -1}}
constexpr int k1 = &arr[1] - &arr[0];
static_assert(k1 == 1, "");
static_assert((&arr[0] - &arr[1]) == -1, "");

constexpr int k2 = &arr2[1] - &arr[0]; // expected-error {{must be initialized by a constant expression}} \
                                       // ref-error {{must be initialized by a constant expression}}

static_assert((arr + 0) == arr, "");
static_assert(&arr[0] == arr, "");
static_assert(*(&arr[0]) == 1, "");
static_assert(*(&arr[1]) == 2, "");

constexpr const int *OOB = (arr + 3) - 3; // expected-error {{must be initialized by a constant expression}} \
                                          // expected-note {{cannot refer to element 3 of array of 2}} \
                                          // ref-error {{must be initialized by a constant expression}} \
                                          // ref-note {{cannot refer to element 3 of array of 2}}

template<typename T>
constexpr T getElementOf(T* array, int i) {
  return array[i];
}
static_assert(getElementOf(foo[0], 1) == &m, "");


template <typename T, int N>
constexpr T& getElementOfArray(T (&array)[N], int I) {
  return array[I];
}
static_assert(getElementOfArray(foo[2], 3) == &m, "");


static_assert(data[0] == 4, ""); // expected-error{{failed}} \
                                 // expected-note{{5 == 4}} \
                                 // ref-error{{failed}} \
                                 // ref-note{{5 == 4}}


constexpr int dynamic[] = {
  f, 3, 2 + 5, data[3], *getElementOf(foo[2], 3)
};
static_assert(dynamic[0] == f, "");
static_assert(dynamic[3] == 2, "");


constexpr int dependent[4] = {
  0, 1, dependent[0], dependent[1]
};
static_assert(dependent[2] == dependent[0], "");
static_assert(dependent[3] == dependent[1], "");

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-extensions"
#pragma clang diagnostic ignored "-Winitializer-overrides"
constexpr int DI[] = {
  [0] = 10,
  [1] = 20,
  30,
  40,
  [1] = 50
};
static_assert(DI[0] == 10, "");
static_assert(DI[1] == 50, "");
static_assert(DI[2] == 30, "");
static_assert(DI[3] == 40, "");

constexpr int addThreeElements(const int v[3]) {
  return v[0] + v[1] + v[2];
}
constexpr int is[] = {10, 20, 30 };
static_assert(addThreeElements(is) == 60, "");

struct fred {
  char s [6];
  int n;
};

struct fred y [] = { [0] = { .s[0] = 'q' } };
#pragma clang diagnostic pop

namespace indices {
  constexpr int first[] = {1};
  constexpr int firstValue = first[2]; // ref-error {{must be initialized by a constant expression}} \
                                       // ref-note {{cannot refer to element 2 of array of 1}} \
                                       // expected-error {{must be initialized by a constant expression}} \
                                       // expected-note {{cannot refer to element 2 of array of 1}}

  constexpr int second[10] = {17};
  constexpr int secondValue = second[10];// ref-error {{must be initialized by a constant expression}} \
                                         // ref-note {{read of dereferenced one-past-the-end pointer}} \
                                         // expected-error {{must be initialized by a constant expression}} \
                                         // expected-note {{read of dereferenced one-past-the-end pointer}}

  constexpr int negative = second[-2]; // ref-error {{must be initialized by a constant expression}} \
                                       // ref-note {{cannot refer to element -2 of array of 10}} \
                                       // expected-error {{must be initialized by a constant expression}} \
                                       // expected-note {{cannot refer to element -2 of array of 10}}
};

namespace DefaultInit {
  template <typename T, unsigned N>
  struct B {
    T a[N];
  };

  int f() {
     constexpr B<int,10> arr = {};
     constexpr int x = arr.a[0];
  }
};

class A {
public:
  int a;
  constexpr A(int m = 2) : a(10 + m) {}
};
class AU {
public:
  int a;
  constexpr AU() : a(5 / 0) {} // expected-warning {{division by zero is undefined}} \
                               // expected-note 2{{division by zero}} \
                               // expected-error {{never produces a constant expression}} \
                               // ref-error {{never produces a constant expression}} \
                               // ref-note 2{{division by zero}} \
                               // ref-warning {{division by zero is undefined}}
};
class B {
public:
  A a[2];
  constexpr B() {}
};
constexpr B b;
static_assert(b.a[0].a == 12, "");
static_assert(b.a[1].a == 12, "");

class BU {
public:
  AU a[2];
  constexpr BU() {} // expected-note {{in call to 'AU()'}} \
                    // ref-note {{in call to 'AU()'}}
};
constexpr BU bu; // expected-error {{must be initialized by a constant expression}} \
                 // expected-note {{in call to 'BU()'}} \
                 // ref-error {{must be initialized by a constant expression}} \
                 // ref-note {{in call to 'BU()'}}

namespace IncDec {
  constexpr int getNextElem(const int *A, int I) {
    const int *B = (A + I);
    ++B;
    return *B;
  }
  constexpr int E[] = {1,2,3,4};

  static_assert(getNextElem(E, 1) == 3, "");

  constexpr int getFirst() {
    const int *e = E;
    return *(e++);
  }
  static_assert(getFirst() == 1, "");

  constexpr int getFirst2() {
    const int *e = E;
    e++;
    return *e;
  }
  static_assert(getFirst2() == 2, "");

  constexpr int getSecond() {
    const int *e = E;
    return *(++e);
  }
  static_assert(getSecond() == 2, "");

  constexpr int getSecond2() {
    const int *e = E;
    ++e;
    return *e;
  }
  static_assert(getSecond2() == 2, "");

  constexpr int getLast() {
    const int *e = E + 3;
    return *(e--);
  }
  static_assert(getLast() == 4, "");

  constexpr int getLast2() {
    const int *e = E + 3;
    e--;
    return *e;
  }
  static_assert(getLast2() == 3, "");

  constexpr int getSecondToLast() {
    const int *e = E + 3;
    return *(--e);
  }
  static_assert(getSecondToLast() == 3, "");

  constexpr int getSecondToLast2() {
    const int *e = E + 3;
    --e;
    return *e;
  }
  static_assert(getSecondToLast2() == 3, "");

  constexpr int bad1() { // ref-error {{never produces a constant expression}} \
                         // expected-error {{never produces a constant expression}}
    const int *e =  E + 3;
    e++; // This is fine because it's a one-past-the-end pointer
    return *e; // expected-note 2{{read of dereferenced one-past-the-end pointer}} \
               // ref-note 2{{read of dereferenced one-past-the-end pointer}}
  }
  static_assert(bad1() == 0, ""); // expected-error {{not an integral constant expression}} \
                                  // expected-note {{in call to}} \
                                  // ref-error {{not an integral constant expression}} \
                                  // ref-note {{in call to}}

  constexpr int bad2() { // ref-error {{never produces a constant expression}} \
                         // expected-error {{never produces a constant expression}}
    const int *e = E + 4;
    e++; // expected-note 2{{cannot refer to element 5 of array of 4 elements}} \
         // ref-note 2{{cannot refer to element 5 of array of 4 elements}}
    return *e; // This is UB as well
  }
  static_assert(bad2() == 0, ""); // expected-error {{not an integral constant expression}} \
                                  // expected-note {{in call to}} \
                                  // ref-error {{not an integral constant expression}} \
                                  // ref-note {{in call to}}


  constexpr int bad3() { // ref-error {{never produces a constant expression}} \
                         // expected-error {{never produces a constant expression}}
    const int *e = E;
    e--; // expected-note 2{{cannot refer to element -1 of array of 4 elements}} \
         // ref-note 2{{cannot refer to element -1 of array of 4 elements}}
    return *e; // This is UB as well
  }
   static_assert(bad3() == 0, ""); // expected-error {{not an integral constant expression}} \
                                   // expected-note {{in call to}} \
                                   // ref-error {{not an integral constant expression}} \
                                  // ref-note {{in call to}}

  constexpr int nullptr1(bool Pre) {
    int *a = nullptr;
    if (Pre)
      ++a; // ref-note {{arithmetic on null pointer}} \
           // expected-note {{arithmetic on null pointer}}
    else
      a++; // ref-note {{arithmetic on null pointer}} \
           // expected-note {{arithmetic on null pointer}}
    return 1;
  }
  static_assert(nullptr1(true) == 1, ""); // ref-error {{not an integral constant expression}} \
                                          // ref-note {{in call to}} \
                                          // expected-error {{not an integral constant expression}} \
                                          // expected-note {{in call to}}

  static_assert(nullptr1(false) == 1, ""); // ref-error {{not an integral constant expression}} \
                                           // ref-note {{in call to}} \
                                           // expected-error {{not an integral constant expression}} \
                                           // expected-note {{in call to}}
};

namespace ZeroInit {
  struct A {
    int *p[2];
  };
  constexpr A a = {};
  static_assert(a.p[0] == nullptr, "");
  static_assert(a.p[1] == nullptr, "");

  struct B {
    double f[2];
  };
  constexpr B b = {};
  static_assert(b.f[0] == 0.0, "");
  static_assert(b.f[1] == 0.0, "");
}

namespace ArrayInitLoop {
  struct X {
      int arr[3];
  };
  constexpr X f(int &r) {
      return {++r, ++r, ++r};
  }
  constexpr int g() {
      int n = 0;
      auto [a, b, c] = f(n).arr;
      return a + b + c;
  }
  static_assert(g() == 6, "");
}

namespace StringZeroFill {
  struct A {
    char c[6];
  };
  constexpr A a = { "abc" };
  static_assert(a.c[0] == 'a', "");
  static_assert(a.c[1] == 'b', "");
  static_assert(a.c[2] == 'c', "");
  static_assert(a.c[3] == '\0', "");
  static_assert(a.c[4] == '\0', "");
  static_assert(a.c[5] == '\0', "");

  constexpr char b[6] = "foo";
  static_assert(b[0] == 'f', "");
  static_assert(b[1] == 'o', "");
  static_assert(b[2] == 'o', "");
  static_assert(b[3] == '\0', "");
  static_assert(b[4] == '\0', "");
  static_assert(b[5] == '\0', "");
}

namespace NoInitMapLeak {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdivision-by-zero"
#pragma clang diagnostic ignored "-Wc++20-extensions"
  constexpr int testLeak() { // expected-error {{never produces a constant expression}} \
                             // ref-error {{never produces a constant expression}}
    int a[2];
    a[0] = 1;
    // interrupts interpretation.
    (void)(1 / 0); // expected-note 2{{division by zero}} \
                   // ref-note 2{{division by zero}}


    return 1;
  }
#pragma clang diagnostic pop
  static_assert(testLeak() == 1, ""); // expected-error {{not an integral constant expression}} \
                                      // expected-note {{in call to 'testLeak()'}} \
                                      // ref-error {{not an integral constant expression}} \
                                      // ref-note {{in call to 'testLeak()'}}


  constexpr int a[] = {1,2,3,4/0,5}; // expected-error {{must be initialized by a constant expression}} \
                                     // expected-note {{division by zero}} \
                                     // ref-error {{must be initialized by a constant expression}} \
                                     // ref-note {{division by zero}} \
                                     // ref-note {{declared here}}

  /// FIXME: This should fail in the new interpreter as well.
  constexpr int b = a[0]; // ref-error {{must be initialized by a constant expression}} \
                          // ref-note {{is not a constant expression}} \
                          // ref-note {{declared here}}
  static_assert(b == 1, ""); // ref-error {{not an integral constant expression}} \
                             // ref-note {{not a constant expression}}

  constexpr int f() { // expected-error {{never produces a constant expression}} \
                      // ref-error {{never produces a constant expression}}
    int a[] = {19,2,3/0,4}; // expected-note 2{{division by zero}} \
                            // expected-warning {{is undefined}} \
                            // ref-note 2{{division by zero}} \
                            // ref-warning {{is undefined}}
    return 1;
  }
  static_assert(f() == 1, ""); // expected-error {{not an integral constant expression}} \
                               // expected-note {{in call to}} \
                               // ref-error {{not an integral constant expression}} \
                               // ref-note {{in call to}}
}

namespace Incomplete {
  struct Foo {
    char c;
    int a[];
  };

  constexpr Foo F{};
  constexpr const int *A = F.a; // ref-error {{must be initialized by a constant expression}} \
                                // ref-note {{array-to-pointer decay of array member without known bound}} \
                                // expected-error {{must be initialized by a constant expression}} \
                                // expected-note {{array-to-pointer decay of array member without known bound}}

  constexpr const int *B = F.a + 1; // ref-error {{must be initialized by a constant expression}} \
                                    // ref-note {{array-to-pointer decay of array member without known bound}} \
                                    // expected-error {{must be initialized by a constant expression}} \
                                    // expected-note {{array-to-pointer decay of array member without known bound}}

  constexpr int C = *F.a; // ref-error {{must be initialized by a constant expression}} \
                          // ref-note {{array-to-pointer decay of array member without known bound}} \
                          // expected-error {{must be initialized by a constant expression}} \
                          // expected-note {{array-to-pointer decay of array member without known bound}}



  /// These are from test/SemaCXX/constant-expression-cxx11.cpp
  /// and are the only tests using the 'indexing of array without known bound' diagnostic.
  /// We currently diagnose them differently.
  extern int arr[]; // expected-note 3{{declared here}}
  constexpr int *c = &arr[1]; // ref-error  {{must be initialized by a constant expression}} \
                              // ref-note {{indexing of array without known bound}} \
                              // expected-error {{must be initialized by a constant expression}} \
                              // expected-note {{read of non-constexpr variable 'arr'}}
  constexpr int *d = &arr[1]; // ref-error  {{must be initialized by a constant expression}} \
                              // ref-note {{indexing of array without known bound}} \
                              // expected-error {{must be initialized by a constant expression}} \
                              // expected-note {{read of non-constexpr variable 'arr'}}
  constexpr int *e = arr + 1; // ref-error  {{must be initialized by a constant expression}} \
                              // ref-note {{indexing of array without known bound}} \
                              // expected-error {{must be initialized by a constant expression}} \
                              // expected-note {{read of non-constexpr variable 'arr'}}
}

namespace GH69115 {
  /// This used to crash because we were trying to emit destructors for the
  /// array.
  constexpr int foo() {
    int arr[2][2] = {1, 2, 3, 4};
    return 0;
  }
  static_assert(foo() == 0, "");

  /// Test that we still emit the destructors for multi-dimensional
  /// composite arrays.
#if __cplusplus >= 202002L
  constexpr void assert(bool C) {
    if (C)
      return;
    // Invalid in constexpr.
    (void)(1 / 0); // expected-warning {{undefined}} \
                   // ref-warning {{undefined}}
  }

  class F {
  public:
    int a;
    int *dtor;
    int &idx;
    constexpr F(int a, int *dtor, int &idx) : a(a), dtor(dtor), idx(idx) {}
    constexpr ~F() noexcept(false){
      dtor[idx] = a;
      ++idx;
    }
  };
  constexpr int foo2() {
    int dtorIndices[] = {0, 0, 0, 0};
    int idx = 0;

    {
      F arr[2][2] = {F(1, dtorIndices, idx),
                     F(2, dtorIndices, idx),
                     F(3, dtorIndices, idx),
                     F(4, dtorIndices, idx)};
    }

    /// Reverse-reverse order.
    assert(idx == 4);
    assert(dtorIndices[0] == 4);
    assert(dtorIndices[1] == 3);
    assert(dtorIndices[2] == 2);
    assert(dtorIndices[3] == 1);

    return 0;
  }
  static_assert(foo2() == 0, "");
#endif
}
