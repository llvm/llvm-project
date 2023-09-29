// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -verify=ref %s

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
  /// FIXME: The ArrayInitLoop for the decomposition initializer in g() has
  /// f(n) as its CommonExpr. We need to evaluate that exactly once and not
  /// N times as we do right now.
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
  static_assert(g() == 6); // expected-error {{failed}} \
                           // expected-note {{15 == 6}}
}
