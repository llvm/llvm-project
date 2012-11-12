// RUN: %clang_cc1 -fsyntax-only %s -verify

// C++'0x [namespace.memdef] p3:
//   Every name first declared in a namespace is a member of that namespace. If
//   a friend declaration in a non-local class first declares a class or
//   function the friend class or function is a member of the innermost
//   enclosing namespace.

namespace N {
  struct S0 {
    friend struct F0;
    friend void f0(int);
    struct F0 member_func();
  };
  struct F0 { };
  F0 f0() { return S0().member_func(); }
}
N::F0 f0_var = N::f0();

// Ensure we can handle attaching friend declarations to an enclosing namespace
// with multiple contexts.
namespace N { struct S1 { struct IS1; }; }
namespace N {
  struct S1::IS1 {
    friend struct F1;
    friend void f1(int);
    struct F1 member_func();
  };
  struct F1 { };
  F1 f1() { return S1::IS1().member_func(); }
}
N::F1 f1_var = N::f1();

//   The name of the friend is not found by unqualified lookup (3.4.1) or by
//   qualified lookup (3.4.3) until a matching declaration is provided in that
//   namespace scope (either before or after the class definition granting
//   friendship). If a friend function is called, its name may be found by the
//   name lookup that considers functions from namespaces and classes
//   associated with the types of the function arguments (3.4.2). If the name
//   in a friend declaration is neither qualified nor a template-id and the
//   declaration is a function or an elaborated-type-specifier, the lookup to
//   determine whether the entity has been previously declared shall not
//   consider any scopes outside the innermost enclosing namespace.

template<typename T> struct X0 { };
struct X1 { };

struct Y {
  template<typename T> union X0;
  template<typename T> friend union X0;
  
  union X1;
  friend union X1;
};

namespace N {
  namespace M {
    template<typename T> class X;
  }
}

namespace N3 {
  class Y {
    template<typename T> friend class N::M::X;
  };
}

// FIXME: Woefully inadequate for testing

// Friends declared as template-ids aren't subject to the restriction
// on innermost namespaces.
// rdar://problem/8552377
namespace test5 {
  template <class T> void f(T);  
  namespace ns {
    class A {
      friend void f<int>(int);
      static void foo(); // expected-note 2 {{declared private here}}
    };

    // Note that this happens without instantiation.
    template <class T> void f(T) {
      A::foo(); // expected-error {{'foo' is a private member of 'test5::ns::A'}}
    }
  }

  template <class T> void f(T) {
    ns::A::foo(); // expected-error {{'foo' is a private member of 'test5::ns::A'}}
  }

  template void f<int>(int);
  template void f<long>(long); //expected-note {{instantiation}}
}
