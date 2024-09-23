// RUN: %clang_cc1 -std=c++2c -fcxx-exceptions -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++2c -fcxx-exceptions -verify=ref,both %s

namespace std {
  using size_t = decltype(sizeof(0));
}

void *operator new(std::size_t, void *p) { return p; }
void* operator new[] (std::size_t, void* p) {return p;}


consteval auto ok1() {
  bool b;
  new (&b) bool(true);
  return b;
}
static_assert(ok1());

consteval auto ok2() {
  int b;
  new (&b) int(12);
  return b;
}
static_assert(ok2() == 12);


consteval auto ok3() {
  float b;
  new (&b) float(12.0);
  return b;
}
static_assert(ok3() == 12.0);


consteval auto ok4() {
  _BitInt(11) b;
  new (&b) _BitInt(11)(37);
  return b;
}
static_assert(ok4() == 37);

/// FIXME: Broken in both interpreters.
#if 0
consteval int ok5() {
    int i;
    new (&i) int[1]{1}; // expected-note {{assignment to dereferenced one-past-the-end pointer}}
    return i;
}
static_assert(ok5() == 1); // expected-error {{not an integral constant expression}} \
                           // expected-note {{in call to}}
#endif

/// FIXME: Crashes the current interpreter.
#if 0
consteval int ok6() {
    int i[2];
    new (&i) int(100);
    return i[0];
}
static_assert(ok6() == 100);
#endif

consteval int ok6() {
    int i[2];
    new (i) int(100);
    new (i + 1) int(200);
    return i[0] + i[1];
}
static_assert(ok6() == 300);


consteval auto fail1() {
  int b;
  new (&b) float(1.0); // both-note {{placement new would change type of storage from 'int' to 'float'}}
  return b;
}
static_assert(fail1() == 0); // both-error {{not an integral constant expression}} \
                             // both-note {{in call to}}

consteval int fail2() {
    int i;
    new (static_cast<void*>(&i)) float(0); // both-note {{placement new would change type of storage from 'int' to 'float'}}
    return 0;
}
static_assert(fail2() == 0); // both-error {{not an integral constant expression}} \
                             // both-note {{in call to}}

consteval int indeterminate() {
    int * indeterminate;
    new (indeterminate) int(0); // both-note {{read of uninitialized object is not allowed in a constant expression}}
    return 0;
}
static_assert(indeterminate() == 0); // both-error {{not an integral constant expression}} \
                                     // both-note {{in call to}}

consteval int array1() {
    int i[2];
    new (&i) int[]{1,2};
    return i[0] + i[1];
}
static_assert(array1() == 3);

consteval int array2() {
    int i[2];
    new (static_cast<void*>(&i)) int[]{1,2};
    return i[0] + i[1];
}
static_assert(array2() == 3);

consteval int array3() {
    int i[1];
    new (&i) int[2]; // both-note {{placement new would change type of storage from 'int[1]' to 'int[2]'}}
    return 0;
}
static_assert(array3() == 0); // both-error {{not an integral constant expression}} \
                              // both-note {{in call to}}

consteval int array4() {
    int i[2];
    new (&i) int[]{12};
    return i[0];
}
static_assert(array4() == 12);

constexpr int *intptr() {
  return new int;
}
constexpr bool yay() {
  int *ptr = new (intptr()) int(42);
  bool ret = *ptr == 42;
  delete ptr;
  return ret;
}
static_assert(yay());


constexpr bool blah() {
  int *ptr = new (intptr()) int[3]{ 1, 2, 3 }; // both-note {{placement new would change type of storage from 'int' to 'int[3]'}}
  bool ret = ptr[0] == 1 && ptr[1] == 2 && ptr[2] == 3;
  delete [] ptr;
  return ret;
}
static_assert(blah()); // both-error {{not an integral constant expression}} \
                       // both-note {{in call to 'blah()'}}


constexpr int *get_indeterminate() {
  int *evil;
  return evil; // both-note {{read of uninitialized object is not allowed in a constant expression}}
}

constexpr bool bleh() {
  int *ptr = new (get_indeterminate()) int;  // both-note {{in call to 'get_indeterminate()'}}
  return true;
}
static_assert(bleh()); // both-error {{not an integral constant expression}} \
                       // both-note {{in call to 'bleh()'}}

namespace records {
  class S {
  public:
    float f;
  };

  constexpr bool record1() {
    S s(13);
    new (&s) S(42);
    return s.f == 42;
  }
  static_assert(record1());

  S GlobalS;
  constexpr bool record2() {
    new (&GlobalS) S(42); // both-note {{a constant expression cannot modify an object that is visible outside that expression}}
    return GlobalS.f == 42;
  }
  static_assert(record2()); // both-error {{not an integral constant expression}} \
                            // both-note {{in call to}}


  constexpr bool record3() {
    S ss[3];

    new (&ss) S[]{{1}, {2}, {3}};

    return ss[0].f == 1 && ss[1].f == 2 && ss[2].f == 3;
  }
  static_assert(record3());

  struct F {
    float f;
  };
  struct R {
    F f;
    int a;
  };
  constexpr bool record4()  {
    R r;
    new (&r.f) F{42.0};
    new (&r.a) int(12);

    return r.f.f == 42.0 && r.a == 12;
  }
  static_assert(record4());

  /// Destructor is NOT called.
  struct A {
    bool b;
    constexpr ~A() { if (b) throw; }
  };

  constexpr int foo() {
    A a;
    new (&a) A(true);
    new (&a) A(false);
    return 0;
  }
  static_assert(foo() == 0);
}
