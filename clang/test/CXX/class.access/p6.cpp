// RUN: %clang_cc1 -fsyntax-only -verify %s

// C++0x [class.access]p6:
//   All access controls in [class.access] affect the ability to
//   access a class member name from a particular scope. For purposes
//   of access control, the base-specifiers of a class and the
//   definitions of class members that appear outside of the class
//   definition are considered to be within the scope of that
//   class. In particular, access controls apply as usual to member
//   names accessed as part of a function return type, even though it
//   is not possible to determine the access privileges of that use
//   without first parsing the rest of the function
//   declarator. Similarly, access control for implicit calls to the
//   constructors, the conversion functions, or the destructor called
//   to create and destroy a static data member is performed as if
//   these calls appeared in the scope of the member's class.

struct Public {}; struct Protected {}; struct Private {};

namespace test0 {
  class A {
    typedef int type; // expected-note {{declared private here}}
    type foo();
  };

  A::type foo() { } // expected-error {{'type' is a private member}}
  A::type A::foo() { }
}

// conversion decls
namespace test1 {
  class A {
  public:
    A();
    operator Public ();
    A(Public);
  protected:
    operator Protected (); // expected-note {{declared protected here}}
    A(Protected); // expected-note {{declared protected here}}
  private:
    operator Private (); // expected-note {{declared private here}}
    A(Private); // expected-note {{declared private here}}
  };

  void test() {
    A a;
    Public pub = a;
    Protected prot = a; // expected-error {{'operator Protected' is a protected member}}
    Private priv = a; // expected-error {{'operator Private' is a private member}}
    A apub = pub;
    A aprot = prot; // expected-error {{protected constructor}}
    A apriv = priv; // expected-error {{private constructor}}
  }
}

// PR6967
namespace test2 {
  class A {
  public:
    template <class T> static void set(T &t, typename T::type v) {
      t.value = v;
    }
    template <class T> static typename T::type get(const T &t) {
      return t.value;
    }
  };

  class B {
    friend class A;

  private:
    typedef int type;
    type value;
  };

  int test() {
    B b;
    A::set(b, 0);
    return A::get(b);
  }
}

namespace test3 {
  class Green {}; class Blue {};

  // We have to wrap this in a class because a partial specialization
  // isn't actually in the context of the template.
  struct Outer {
    template <class T, class Nat> class A {
    };
  };

  template <class T> class Outer::A<T, typename T::nature> {
  public:
    static void foo(); // expected-note {{'Outer::A<B, Green>::foo' declared here}}
  };

  class B {
  private: typedef Green nature;
    friend class Outer;
  };

  void test() {
    Outer::A<B, Green>::foo();
    Outer::A<B, Blue>::foo(); // expected-error {{no member named 'foo' in 'test3::Outer::A<test3::B, test3::Blue>'; did you mean 'Outer::A<B, Green>::foo'?}}
  }
}

namespace test4 {
  template <class T> class A {
  private: typedef int type;
    template <class U> friend void foo(U &, typename U::type);
  };

  template <class U> void foo(U &, typename U::type) {}
  
  void test() {
    A<int> a;
    foo(a, 0);
  }
}

// PR7644
namespace test5 {
  class A {
    enum Enum { E0, E1, E2 }; // expected-note 4 {{declared private here}}
    template <Enum> void foo();
    template <Enum> class bar;
  };

  template <A::Enum en> void A::foo() {}
  template <A::Enum en> class A::bar {};

  template <A::Enum en> void foo() {} // expected-error {{'Enum' is a private member of 'test5::A'}}
  template <A::Enum en> class bar {}; // expected-error {{'Enum' is a private member of 'test5::A'}}

  class B {
    template <A::Enum en> void foo() {} // expected-error {{'Enum' is a private member of 'test5::A'}}
    template <A::Enum en> class bar {}; // expected-error {{'Enum' is a private member of 'test5::A'}}
  };
}

namespace test6 {
  class A {
  public: class public_inner {};
  protected: class protected_inner {};
  private: class private_inner {}; // expected-note {{declared private here}}
  };

  class B : A {
    public_inner a;
    protected_inner b;
    private_inner c; // expected-error {{'private_inner' is a private member of 'test6::A'}}
  };
}

// PR9229
namespace test7 {
  void foo(int arg[1]);
  class A {
    void check();
  };
  class B {
    friend class A;
    A ins;
  };
  void A::check() {
    void foo(int arg[__builtin_offsetof(B, ins)]);
  }
}

// rdar://problem/10155256
namespace test8 {
  class A {
    typedef void* (A::*UnspecifiedBoolType)() const;
    operator UnspecifiedBoolType() const; // expected-note {{implicitly declared private here}}
  };

  void test(A &a) {
    if (a) return; // expected-error-re {{'operator void *(test8::A::*)(){{( __attribute__\(\(thiscall\)\))?}} const' is a private member of 'test8::A'}}
  }
}

namespace test9 {
  class A {
    operator char*() const; // expected-note {{implicitly declared private here}}
  };

  void test(A &a) {
    delete a; // expected-error {{'operator char *' is a private member of 'test9::A'}}
  }
}
