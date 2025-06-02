// RUN: %clang_cc1 -std=c++2c -fcxx-exceptions -fexperimental-new-constant-interpreter -verify=expected,both %s -DBYTECODE
// RUN: %clang_cc1 -std=c++2c -fcxx-exceptions -verify=ref,both %s

namespace std {
  using size_t = decltype(sizeof(0));
  template<typename T> struct allocator {
    constexpr T *allocate(size_t N) {
      return (T*)operator new(sizeof(T) * N);
    }
    constexpr void deallocate(void *p) {
      operator delete(p);
    }
  };
  template<typename T, typename ...Args>
  constexpr void construct_at(void *p, Args &&...args) {
    new (p) T((Args&&)args...); // both-note {{in call to}} \
                                // both-note {{placement new would change type of storage from 'int' to 'float'}} \
                                // both-note {{construction of subobject of member 'x' of union with active member 'a' is not allowed in a constant expression}} \
                                // both-note {{construction of temporary is not allowed}} \
                                // both-note {{construction of heap allocated object that has been deleted}} \
                                // both-note {{construction of subobject of object outside its lifetime is not allowed in a constant expression}}
  }
}

void *operator new(std::size_t, void *p) { return p; }
void* operator new[] (std::size_t, void* p) {return p;}

constexpr int no_lifetime_start = (*std::allocator<int>().allocate(1) = 1); // both-error {{constant expression}} \
                                                                            // both-note {{assignment to object outside its lifetime}}

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

consteval int ok5() {
  int i;
  new (&i) int[1]{1};

  struct S {
    int a; int b;
  } s;
  new (&s) S[1]{{12, 13}};

  return 25;
  // return s.a + s.b; FIXME: Broken in the current interpreter.
}
static_assert(ok5() == 25);

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

namespace ConstructAt {
  struct S {
    int a = 10;
    float b = 1.0;
  };

  constexpr bool ok1() {
    S s;

    std::construct_at<S>(&s);
    return s.a == 10 && s.b == 1.0;
  }
  static_assert(ok1());

  struct S2 {
    constexpr S2() {
      (void)(1/0); // both-note {{division by zero}} \
                   // both-warning {{division by zero is undefined}}
    }
  };

  constexpr bool ctorFail() { //
    S2 *s = std::allocator<S2>().allocate(1);
    std::construct_at<S2>(s); // both-note {{in call to}}

    return true;
  }
  static_assert(ctorFail()); // both-error {{not an integral constant expression}} \
                             // both-note {{in call to 'ctorFail()'}}


  constexpr bool bad_construct_at_type() {
    int a;
    std::construct_at<float>(&a, 1.0f); // both-note {{in call to}}
    return true;
  }
  static_assert(bad_construct_at_type()); // both-error {{not an integral constant expression}} \
                                          // both-note {{in call}}

  constexpr bool bad_construct_at_subobject() {
    struct X { int a, b; };
    union A {
      int a;
      X x;
    };
    A a = {1};
    std::construct_at<int>(&a.x.a, 1); // both-note {{in call}}
    return true;
  }
  static_assert(bad_construct_at_subobject()); // both-error{{not an integral constant expression}} \
                                               // both-note {{in call}}
}

namespace UsedToCrash {
  struct S {
      int* i;
      constexpr S() : i(new int(42)) {} // #no-deallocation
      constexpr ~S() {delete i;}
  };
  consteval void alloc() {
      S* s = new S();
      s->~S();
      new (s) S();
      delete s;
  }
  int alloc1 = (alloc(), 0);
}

constexpr bool change_union_member() {
  union U {
    int a;
    int b;
  };
  U u = {.a = 1};
  std::construct_at<int>(&u.b, 2);
  return u.b == 2;
}
static_assert(change_union_member());

namespace PR48606 {
  struct A { mutable int n = 0; };

  constexpr bool f() {
    A a;
    A *p = &a;
    p->~A();
    std::construct_at<A>(p);
    return true;
  }
  static_assert(f());
}

/// This used to crash because of an assertion in the implementation
/// of the This instruction.
namespace ExplicitThisOnArrayElement {
  struct S {
    int a = 12;
    constexpr S(int a) {
      this->a = a;
    }
  };

  template <class _Tp, class... _Args>
  constexpr void construct_at(_Tp *__location, _Args &&...__args) {
    new (__location) _Tp(__args...);
  }

  constexpr bool foo() {
    auto *M = std::allocator<S>().allocate(13); // both-note {{allocation performed here was not deallocated}}
    construct_at(M, 12);
    return true;
  }

  static_assert(foo()); // both-error {{not an integral constant expression}}
}

#ifdef BYTECODE
constexpr int N = [] // expected-error {{must be initialized by a constant expression}} \
                     // expected-note {{assignment to dereferenced one-past-the-end pointer is not allowed in a constant expression}} \
                     // expected-note {{in call to}}
{
    struct S {
        int a[1];
    };
    S s;
    ::new (s.a) int[1][2][3][4]();
    return s.a[0];
}();
#endif

namespace MemMove {
  constexpr int foo() {
    int *a = std::allocator<int>{}.allocate(1);
    new(a) int{123};

    int b;
    __builtin_memmove(&b, a, sizeof(int));

    std::allocator<int>{}.deallocate(a);
    return b;
  }

  static_assert(foo() == 123);
}

namespace Temp {
  constexpr int &&temporary = 0; // both-note {{created here}}
  static_assert((std::construct_at<int>(&temporary, 1), true)); // both-error{{not an integral constant expression}} \
                                                                // both-note {{in call}}
}

namespace PlacementNewAfterDelete {
  constexpr bool construct_after_lifetime() {
    int *p = new int;
    delete p;
    std::construct_at<int>(p); // both-note {{in call}}
    return true;
  }
  static_assert(construct_after_lifetime()); // both-error {{}} \
                                             // both-note {{in call}}
}

namespace SubObj {
  constexpr bool construct_after_lifetime_2() {
    struct A { struct B {} b; };
    A a;
    a.~A();
    std::construct_at<A::B>(&a.b); // both-note {{in call}}
    return true;
  }
  static_assert(construct_after_lifetime_2()); // both-error {{}} both-note {{in call}}
}
