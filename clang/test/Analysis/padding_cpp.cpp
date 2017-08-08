// RUN: %clang_analyze_cc1 -std=c++14 -analyzer-checker=optin.performance -analyzer-config optin.performance.Padding:AllowedPad=2 -verify %s

// Make sure that the C cases still work fine, even when compiled as C++.
#include "padding_c.c"

struct BigCharArray2 { // no-warning
  char c[129];
};

// xxxexpected-warning@+1{{Excessive padding in 'struct LowAlignmentBase'}}
struct LowAlignmentBase : public BigCharArray2 {
  int i;
  char c;
};

struct CorrectLowAlignmentBase : public BigCharArray2 { // no-warning
  char c;
  int i;
};

// xxxexpected-warning@+1{{Excessive padding in 'struct LowAlignmentBase2'}}
struct LowAlignmentBase2 : public BigCharArray2 {
  char c1;
  int i;
  char c2;
};

class PaddedA { // expected-warning{{Excessive padding in 'class PaddedA'}}
  char c1;
  int i;
  char c2;
};

class VirtualPaddedA : public PaddedA { // no-warning
  virtual void foo() {}
};

class VirtualIntSandwich { // expected-warning{{Excessive padding in 'class VirtualIntSandwich'}}
  virtual void foo() {}
  char c1;
  int i;
  char c2;
};

// constructed so as not to have tail padding
class InnerPaddedB { // expected-warning{{Excessive padding in 'class InnerPaddedB'}}
  char c1;
  int i1;
  char c2;
  int i2;
};

class TailPaddedB { // expected-warning{{Excessive padding in 'class TailPaddedB'}}
  char c1;
  int i1;
  char c2;
};

class SI : public PaddedA { // no-warning
  char c;
};

class SI2 : public PaddedA { // xxxexpected-warning{{Excessive padding in 'class SI2'}}
  char c10;
  int i10;
  char c11;
};

class VirtualSI : virtual public PaddedA { // no-warning
  char c;
};

// currently not checked for
class VirtualSI2 : virtual public PaddedA { // no-warning
  char c10;
  int i10;
  char c11;
};

class VtblSI : public PaddedA { // no-warning
  virtual void foo() {}
  char c;
};

class VtblSI2 : public PaddedA { // xxxexpected-warning{{Excessive padding in 'class VtblSI2'}}
  virtual void foo() {}
  char c10;
  int i10;
  char c11;
};

class VtblSI3 : public VirtualPaddedA { // xxxexpected-warning{{Excessive padding in 'class VtblSI3'}}
  char c10;
  int i10;
  char c11;
};

class MI : public PaddedA, public InnerPaddedB { // no-warning
  char c;
};

class MI2 : public PaddedA, public InnerPaddedB { // xxxexpected-warning{{Excessive padding in 'class MI2'}}
  char c10;
  int i10;
  char c11;
};

class VtblMI : public PaddedA, public InnerPaddedB { // xxxexpected-warning{{Excessive padding in 'class VtblMI'}}
  virtual void foo() {}
  char c10;
  int i10;
  char c11;
};

class VtblMI2 : public VirtualPaddedA, public InnerPaddedB { // xxxexpected-warning{{Excessive padding in 'class VtblMI2'}}
  char c10;
  int i10;
  char c11;
};

class Empty {}; // no-warning

class LotsOfSpace { // expected-warning{{Excessive padding in 'class LotsOfSpace'}}
  Empty e1;
  int i;
  Empty e2;
};

class EBO1 : public Empty { // xxxexpected-warning{{Excessive padding in 'class EBO1'}}
  char c1;
  int i;
  char c2;
};

class EBO2 : public Empty { // xxxexpected-warning{{Excessive padding in 'class EBO2'}}
  Empty c1;
  int i;
  Empty c2;
};

template <typename T>
class TemplateSandwich { // expected-warning{{Excessive padding in 'class TemplateSandwich<int>' instantiated here}}
  char c1;
  T t;
  char c2;
};

template <typename T>
class TemplateSandwich<T *> { // expected-warning{{Excessive padding in 'class TemplateSandwich<void *>' instantiated here}}
  char c1;
  T *t;
  char c2;
};

template <>
class TemplateSandwich<long long> { // expected-warning{{Excessive padding in 'class TemplateSandwich<long long>' (}}
  char c1;
  long long t;
  char c2;
};

class Holder1 { // no-warning
  TemplateSandwich<int> t1;
  TemplateSandwich<char> t2;
  TemplateSandwich<void *> t3;
};

typedef struct { // expected-warning{{Excessive padding in 'TypedefSandwich2'}}
  char c1;
  typedef struct { // expected-warning{{Excessive padding in 'TypedefSandwich2::NestedTypedef'}}
    char c1;
    int i;
    char c2;
  } NestedTypedef;
  NestedTypedef t;
  char c2;
} TypedefSandwich2;

template <typename T>
struct Foo {
  // expected-warning@+1{{Excessive padding in 'struct Foo<int>::Nested'}}
  struct Nested {
    char c1;
    T t;
    char c2;
  };
};

struct Holder { // no-warning
  Foo<int>::Nested t1;
  Foo<char>::Nested t2;
};

struct GlobalsForLambda { // no-warning
  int i;
  char c1;
  char c2;
} G;

// expected-warning@+1{{Excessive padding in 'class (lambda}}
auto lambda1 = [ c1 = G.c1, i = G.i, c2 = G.c2 ]{};
auto lambda2 = [ i = G.i, c1 = G.c1, c2 = G.c2 ]{}; // no-warning
