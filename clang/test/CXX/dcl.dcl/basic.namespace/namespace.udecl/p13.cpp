// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// C++03 [namespace.udecl]p3:
//   For the purpose of overload resolution, the functions which are
//   introduced by a using-declaration into a derived class will be
//   treated as though they were members of the derived class. In
//   particular, the implicit this parameter shall be treated as if it
//   were a pointer to the derived class rather than to the base
//   class. This has no effect on the type of the function, and in all
//   other respects the function remains a member of the base class.

namespace test0 {
  struct Opaque0 {};
  struct Opaque1 {};

  struct Base {
    Opaque0 test0(int*);
    Opaque0 test1(const int*);
    Opaque0 test2(int*);
    Opaque0 test3(int*) const;
  };

  struct Derived : Base {
    using Base::test0;
    Opaque1 test0(const int*);

    using Base::test1;
    Opaque1 test1(int*);

    using Base::test2;
    Opaque1 test2(int*) const;

    using Base::test3;
    Opaque1 test3(int*);
  };

  void test0() {
    Opaque0 a = Derived().test0((int*) 0);
    Opaque1 b = Derived().test0((const int*) 0);
  }

  void test1() {
    Opaque1 a = Derived().test1((int*) 0);
    Opaque0 b = Derived().test1((const int*) 0);
  }

  void test2() {
    Opaque0 a = ((Derived*) 0)->test2((int*) 0);
    Opaque1 b = ((const Derived*) 0)->test2((int*) 0);
  }

  void test3() {
    Opaque1 a = ((Derived*) 0)->test3((int*) 0);
    Opaque0 b = ((const Derived*) 0)->test3((int*) 0);
  }
}

// Typedef redeclaration.
namespace rdar8018262 {
  typedef void (*fp)();

  namespace N {
    typedef void (*fp)();
  }

  using N::fp;

  fp fp_1;
}

// Things to test:
//   member operators
//   conversion operators
//   call operators
//   call-surrogate conversion operators
//   everything, but in dependent contexts
