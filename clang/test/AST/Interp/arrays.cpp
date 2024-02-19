// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++20 -verify=expected,both %s
// RUN: %clang_cc1 -verify=ref,both %s
// RUN: %clang_cc1 -verify=ref,both -std=c++20 %s

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

constexpr int getFirstElem(const int *a) {
  return a[0]; // both-note {{read of dereferenced null pointer}}
}
static_assert(getFirstElem(nullptr) == 1, ""); // both-error {{not an integral constant expression}} \
                                               // both-note {{in call to}}

constexpr static int arr[2] = {1,2};
constexpr static int arr2[2] = {3,4};
constexpr int *p1 = nullptr;
constexpr int *p2 = p1 + 1; // both-error {{must be initialized by a constant expression}} \
                            // both-note {{cannot perform pointer arithmetic on null pointer}}
constexpr int *p3 = p1 + 0;
constexpr int *p4 = p1 - 0;
constexpr int *p5 =  0 + p1;
constexpr int *p6 =  0 - p1; // both-error {{invalid operands to binary expression}}

constexpr int const * ap1 = &arr[0];
constexpr int const * ap2 = ap1 + 3; // both-error {{must be initialized by a constant expression}} \
                                     // both-note {{cannot refer to element 3 of array of 2}}

constexpr auto ap3 = arr - 1; // both-error {{must be initialized by a constant expression}} \
                              // both-note {{cannot refer to element -1}}
constexpr int k1 = &arr[1] - &arr[0];
static_assert(k1 == 1, "");
static_assert((&arr[0] - &arr[1]) == -1, "");

constexpr int k2 = &arr2[1] - &arr[0]; // both-error {{must be initialized by a constant expression}}

static_assert((arr + 0) == arr, "");
static_assert(&arr[0] == arr, "");
static_assert(*(&arr[0]) == 1, "");
static_assert(*(&arr[1]) == 2, "");

constexpr const int *OOB = (arr + 3) - 3; // both-error {{must be initialized by a constant expression}} \
                                          // both-note {{cannot refer to element 3 of array of 2}}

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


static_assert(data[0] == 4, ""); // both-error{{failed}} \
                                 // both-note{{5 == 4}}

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
  constexpr int firstValue = first[2]; // both-error {{must be initialized by a constant expression}} \
                                       // both-note {{cannot refer to element 2 of array of 1}}

  constexpr int second[10] = {17};
  constexpr int secondValue = second[10];// both-error {{must be initialized by a constant expression}} \
                                         // both-note {{read of dereferenced one-past-the-end pointer}} \

  constexpr int negative = second[-2]; // both-error {{must be initialized by a constant expression}} \
                                       // both-note {{cannot refer to element -2 of array of 10}}
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
  constexpr AU() : a(5 / 0) {} // both-warning {{division by zero is undefined}} \
                               // both-note 2{{division by zero}} \
                               // both-error {{never produces a constant expression}}
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
  constexpr BU() {} // both-note {{in call to 'AU()'}}
};
constexpr BU bu; // both-error {{must be initialized by a constant expression}} \
                 // both-note {{in call to 'BU()'}}

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

  constexpr int bad1() { // both-error {{never produces a constant expression}}
    const int *e =  E + 3;
    e++; // This is fine because it's a one-past-the-end pointer
    return *e; // both-note 2{{read of dereferenced one-past-the-end pointer}}
  }
  static_assert(bad1() == 0, ""); // both-error {{not an integral constant expression}} \
                                  // both-note {{in call to}}

  constexpr int bad2() { // both-error {{never produces a constant expression}}
    const int *e = E + 4;
    e++; // both-note 2{{cannot refer to element 5 of array of 4 elements}}
    return *e; // This is UB as well
  }
  static_assert(bad2() == 0, ""); // both-error {{not an integral constant expression}} \
                                  // both-note {{in call to}}

  constexpr int bad3() { // both-error {{never produces a constant expression}}
    const int *e = E;
    e--; // both-note 2{{cannot refer to element -1 of array of 4 elements}}
    return *e; // This is UB as well
  }
   static_assert(bad3() == 0, ""); // both-error {{not an integral constant expression}} \
                                   // both-note {{in call to}}

  constexpr int nullptr1(bool Pre) {
    int *a = nullptr;
    if (Pre)
      ++a; // both-note {{arithmetic on null pointer}}
    else
      a++; // both-note {{arithmetic on null pointer}}
    return 1;
  }
  static_assert(nullptr1(true) == 1, ""); // both-error {{not an integral constant expression}} \
                                          // both-note {{in call to}}

  static_assert(nullptr1(false) == 1, ""); // both-error {{not an integral constant expression}} \
                                           // both-note {{in call to}}
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
  constexpr int testLeak() { // both-error {{never produces a constant expression}}
    int a[2];
    a[0] = 1;
    // interrupts interpretation.
    (void)(1 / 0); // both-note 2{{division by zero}}

    return 1;
  }
#pragma clang diagnostic pop
  static_assert(testLeak() == 1, ""); // both-error {{not an integral constant expression}} \
                                      // both-note {{in call to 'testLeak()'}}

  constexpr int a[] = {1,2,3,4/0,5}; // both-error {{must be initialized by a constant expression}} \
                                     // both-note {{division by zero}} \
                                     // ref-note {{declared here}}

  /// FIXME: This should fail in the new interpreter as well.
  constexpr int b = a[0]; // ref-error {{must be initialized by a constant expression}} \
                          // ref-note {{is not a constant expression}} \
                          // ref-note {{declared here}}
  static_assert(b == 1, ""); // ref-error {{not an integral constant expression}} \
                             // ref-note {{not a constant expression}}

  constexpr int f() { // both-error {{never produces a constant expression}}
    int a[] = {19,2,3/0,4}; // both-note 2{{division by zero}} \
                            // both-warning {{is undefined}}
    return 1;
  }
  static_assert(f() == 1, ""); // both-error {{not an integral constant expression}} \
                               // both-note {{in call to}}
}

namespace Incomplete {
  struct Foo {
    char c;
    int a[];
  };

  constexpr Foo F{};
  constexpr const int *A = F.a; // both-error {{must be initialized by a constant expression}} \
                                // both-note {{array-to-pointer decay of array member without known bound}}

  constexpr const int *B = F.a + 1; // both-error {{must be initialized by a constant expression}} \
                                    // both-note {{array-to-pointer decay of array member without known bound}}

  constexpr int C = *F.a; // both-error {{must be initialized by a constant expression}} \
                          // both-note {{array-to-pointer decay of array member without known bound}}

  /// These are from test/SemaCXX/constant-expression-cxx11.cpp
  /// and are the only tests using the 'indexing of array without known bound' diagnostic.
  /// We currently diagnose them differently.
  extern int arr[]; // expected-note 3{{declared here}}
  constexpr int *c = &arr[1]; // both-error  {{must be initialized by a constant expression}} \
                              // ref-note {{indexing of array without known bound}} \
                              // expected-note {{read of non-constexpr variable 'arr'}}
  constexpr int *d = &arr[1]; // both-error  {{must be initialized by a constant expression}} \
                              // ref-note {{indexing of array without known bound}} \
                              // expected-note {{read of non-constexpr variable 'arr'}}
  constexpr int *e = arr + 1; // both-error  {{must be initialized by a constant expression}} \
                              // ref-note {{indexing of array without known bound}} \
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
    (void)(1 / 0); // both-warning {{undefined}}
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

namespace NonConstReads {
#if __cplusplus >= 202002L
  void *p = nullptr; // both-note {{declared here}}

  int arr[!p]; // both-error {{not allowed at file scope}} \
               // both-warning {{variable length arrays}} \
               // both-note {{read of non-constexpr variable 'p'}}
  int z; // both-note {{declared here}}
  int a[z]; // both-error {{not allowed at file scope}} \
            // both-warning {{variable length arrays}} \
            // both-note {{read of non-const variable 'z'}}
#else
  void *p = nullptr;
  int arr[!p]; // ref-error {{not allowed at file scope}} \
               // expected-error {{not allowed at file scope}}
  int z;
  int a[z]; // ref-error {{not allowed at file scope}} \
            // expected-error {{not allowed at file scope}}
#endif

  const int y = 0;
  int yy[y];
}

namespace SelfComparison {
  struct S {
    int field;
    static int static_field;
    int array[4];
  };

  struct T {
    int field;
    static int static_field;
    int array[4];
    S s;
  };

  int struct_test(S s1, S s2, S *s3, T t) {
    return s3->array[t.field] == s3->array[t.field];  // both-warning {{self-comparison always evaluates to true}}
  };
}

namespace LocalIndex {
  void test() {
    const int const_subscript = 3;
    int array[2]; // both-note {{declared here}}
    array[const_subscript] = 0;  // both-warning {{array index 3 is past the end of the array (that has type 'int[2]')}}
  }
}

namespace LocalVLA {
  struct Foo {
    int x;
    Foo(int x) : x(x) {}
  };
  struct Elidable {
    Elidable();
  };

  void foo(int size) {
    Elidable elidableDynArray[size];
#if __cplusplus >= 202002L
     // both-note@-3 {{declared here}}
     // both-warning@-3 {{variable length array}}
     // both-note@-4 {{function parameter 'size' with unknown value}}
#endif
  }
}
