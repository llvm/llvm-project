// RUN: %clang_cc1 -std=c++2b -fsyntax-only -fcxx-exceptions                       -verify=expected,cxx11_2b,cxx2b    %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fcxx-exceptions                       -verify=expected,cxx98_20,cxx11_2b,cxx11_20 %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fcxx-exceptions                       -verify=expected,cxx98_20,cxx11_2b,cxx11_20 %s
// RUN: %clang_cc1 -std=c++98 -fsyntax-only -fcxx-exceptions -Wno-c++11-extensions -verify=expected,cxx98_20,cxx98 %s

namespace test_delete_function {
struct A1 {
  A1();
  A1(const A1 &);
  A1(A1 &&) = delete; // expected-note {{'A1' has been explicitly marked deleted here}}
};
A1 test1() {
  A1 a;
  return a; // expected-error {{call to deleted constructor of 'test_delete_function::A1'}}
}

struct A2 {
  A2();
  A2(const A2 &);

private:
  A2(A2 &&); // expected-note {{declared private here}}
};
A2 test2() {
  A2 a;
  return a; // expected-error {{calling a private constructor of class 'test_delete_function::A2'}}
}

struct C {};

struct B1 {
  B1(C &);
  B1(C &&) = delete; // expected-note {{'B1' has been explicitly marked deleted here}}
};
B1 test3() {
  C c;
  return c; // expected-error {{conversion function from 'test_delete_function::C' to 'test_delete_function::B1' invokes a deleted function}}
}

struct B2 {
  B2(C &);

private:
  B2(C &&); // expected-note {{declared private here}}
};
B2 test4() {
  C c;
  return c; // expected-error {{calling a private constructor of class 'test_delete_function::B2'}}
}
} // namespace test_delete_function

// Implicitly movable entity can be rvalue reference to non-volatile
// automatic object.
namespace test_implicitly_movable_rvalue_ref {
struct A1 {
  A1(A1 &&);
  A1(const A1 &) = delete;
};
A1 test1(A1 &&a) {
  return a;
}

struct A2 {
  A2(A2 &&);

private:
  A2(const A2 &);
};
A2 test2(A2 &&a) {
  return a;
}

struct B1 {
  B1(const B1 &);
  B1(B1 &&) = delete; // expected-note {{'B1' has been explicitly marked deleted here}}
};
B1 test3(B1 &&b) {
  return b; // expected-error {{call to deleted constructor of 'test_implicitly_movable_rvalue_ref::B1'}}
}

struct B2 {
  B2(const B2 &);

private:
  B2(B2 &&); // expected-note {{declared private here}}
};
B2 test4(B2 &&b) {
  return b; // expected-error {{calling a private constructor of class 'test_implicitly_movable_rvalue_ref::B2'}}
}
} // namespace test_implicitly_movable_rvalue_ref

// Operand of throw-expression can be function parameter or
// catch-clause parameter.
namespace test_throw_parameter {
void func();

struct A1 {
  A1(const A1 &);
  A1(A1 &&) = delete; // expected-note 2{{'A1' has been explicitly marked deleted here}}
};
void test1() {
  try {
    func();
  } catch (A1 a) {
    throw a; // expected-error {{call to deleted constructor of 'test_throw_parameter::A1'}}
  }
}

struct A2 {
  A2(const A2 &);

private:
  A2(A2 &&); // expected-note {{declared private here}}
};
void test2() {
  try {
    func();
  } catch (A2 a) {
    throw a; // expected-error {{calling a private constructor of class 'test_throw_parameter::A2'}}
  }
}

void test3(A1 a) try {
  func();
} catch (...) {
  throw a; // expected-error {{call to deleted constructor of 'test_throw_parameter::A1'}}
}
} // namespace test_throw_parameter

// During the first overload resolution, the selected function no
// need to be a constructor.
namespace test_non_ctor_conversion {
class C {};

struct A1 {
  operator C() &&;
  operator C() const & = delete;
};
C test1() {
  A1 a;
  return a;
}

struct A2 {
  operator C() &&;

private:
  operator C() const &;
};
C test2() {
  A2 a;
  return a;
}

struct B1 {
  operator C() const &;
  operator C() && = delete; // expected-note {{'operator C' has been explicitly marked deleted here}}
};
C test3() {
  B1 b;
  return b; // expected-error {{conversion function from 'test_non_ctor_conversion::B1' to 'test_non_ctor_conversion::C' invokes a deleted function}}
}

struct B2 {
  operator C() const &;

private:
  operator C() &&; // expected-note {{declared private here}}
};
C test4() {
  B2 b;
  return b; // expected-error {{'operator C' is a private member of 'test_non_ctor_conversion::B2'}}
}
} // namespace test_non_ctor_conversion

// During the first overload resolution, the first parameter of the
// selected function no need to be an rvalue reference to the object's type.
namespace test_ctor_param_rvalue_ref {
struct A1;
struct A2;
struct B1;
struct B2;

struct NeedRvalueRef {
  NeedRvalueRef(A1 &&);
  NeedRvalueRef(A2 &&);
  NeedRvalueRef(B1 &&);
  NeedRvalueRef(B2 &&);
};
struct NeedValue {
  NeedValue(A1); // cxx98-note 2 {{passing argument to parameter here}}
  NeedValue(A2);
  NeedValue(B1); // cxx11_2b-note 2 {{passing argument to parameter here}}
  NeedValue(B2);
};

struct A1 {
  A1();
  A1(A1 &&);
  A1(const A1 &) = delete; // cxx98-note 2 {{marked deleted here}}
};
NeedValue test_1_1() {
  // not rvalue reference
  // same type
  A1 a;
  return a; // cxx98-error {{call to deleted constructor}}
}
class DerivedA1 : public A1 {};
A1 test_1_2() {
  // rvalue reference
  // not same type
  DerivedA1 a;
  return a;
}
NeedValue test_1_3() {
  // not rvalue reference
  // not same type
  DerivedA1 a;
  return a; // cxx98-error {{call to deleted constructor}}
}

struct A2 {
  A2();
  A2(A2 &&);

private:
  A2(const A2 &); // cxx98-note 2 {{declared private here}}
};
NeedValue test_2_1() {
  // not rvalue reference
  // same type
  A2 a;
  return a; // cxx98-error {{calling a private constructor}}
}
class DerivedA2 : public A2 {};
A2 test_2_2() {
  // rvalue reference
  // not same type
  DerivedA2 a;
  return a;
}
NeedValue test_2_3() {
  // not rvalue reference
  // not same type
  DerivedA2 a;
  return a; // cxx98-error {{calling a private constructor}}
}

struct B1 {
  B1();
  B1(const B1 &);
  B1(B1 &&) = delete; // cxx11_2b-note 3 {{'B1' has been explicitly marked deleted here}}
                      // cxx98-note@-1 {{'B1' has been explicitly marked deleted here}}
};
NeedValue test_3_1() {
  // not rvalue reference
  // same type
  B1 b;
  return b; // cxx11_2b-error {{call to deleted constructor of 'test_ctor_param_rvalue_ref::B1'}}
}
class DerivedB1 : public B1 {};
B1 test_3_2() {
  // rvalue reference
  // not same type
  DerivedB1 b;
  return b; // expected-error {{call to deleted constructor of 'test_ctor_param_rvalue_ref::B1'}}
}
NeedValue test_3_3() {
  // not rvalue reference
  // not same type
  DerivedB1 b;
  return b; // cxx11_2b-error {{call to deleted constructor of 'test_ctor_param_rvalue_ref::B1'}}
}

struct B2 {
  B2();
  B2(const B2 &);

private:
  B2(B2 &&); // cxx11_2b-note 3 {{declared private here}}
             // cxx98-note@-1 {{declared private here}}
};
NeedValue test_4_1() {
  // not rvalue reference
  // same type
  B2 b;
  return b; // cxx11_2b-error {{calling a private constructor of class 'test_ctor_param_rvalue_ref::B2'}}
}
class DerivedB2 : public B2 {};
B2 test_4_2() {
  // rvalue reference
  // not same type
  DerivedB2 b;
  return b; // expected-error {{calling a private constructor of class 'test_ctor_param_rvalue_ref::B2'}}
}
NeedValue test_4_3() {
  // not rvalue reference
  // not same type
  DerivedB2 b;
  return b; // cxx11_2b-error {{calling a private constructor of class 'test_ctor_param_rvalue_ref::B2'}}
}
} // namespace test_ctor_param_rvalue_ref

namespace test_lvalue_ref_is_not_moved_from {

struct Target {};
// expected-note@-1  {{candidate constructor (the implicit copy constructor) not viable}}
// cxx11_2b-note@-2  {{candidate constructor (the implicit move constructor) not viable}}

struct CopyOnly {
  CopyOnly(CopyOnly &&) = delete; // expected-note {{has been explicitly marked deleted here}}
  CopyOnly(CopyOnly&);
  operator Target() && = delete; // expected-note {{has been explicitly marked deleted here}}
  operator Target() &;
};

struct MoveOnly {
  MoveOnly(MoveOnly &&); // cxx11_2b-note {{copy constructor is implicitly deleted because}}
  operator Target() &&;  // expected-note {{candidate function not viable}}
};

extern CopyOnly copyonly;
extern MoveOnly moveonly;

CopyOnly t1() {
    CopyOnly& r = copyonly;
    return r;
}

CopyOnly t2() {
    CopyOnly&& r = static_cast<CopyOnly&&>(copyonly);
    return r; // expected-error {{call to deleted constructor}}
}

MoveOnly t3() {
    MoveOnly& r = moveonly;
    return r; // cxx11_2b-error {{call to implicitly-deleted copy constructor}}
}

MoveOnly t4() {
    MoveOnly&& r = static_cast<MoveOnly&&>(moveonly);
    return r;
}

Target t5() {
    CopyOnly& r = copyonly;
    return r;
}

Target t6() {
    CopyOnly&& r = static_cast<CopyOnly&&>(copyonly);
    return r; // expected-error {{invokes a deleted function}}
}

Target t7() {
    MoveOnly& r = moveonly;
    return r; // expected-error {{no viable conversion}}
}

Target t8() {
    MoveOnly&& r = static_cast<MoveOnly&&>(moveonly);
    return r;
}

} // namespace test_lvalue_ref_is_not_moved_from

namespace test_rvalue_ref_to_nonobject {

struct CopyOnly {};
struct MoveOnly {};

struct Target {
    Target(CopyOnly (&)());
    Target(CopyOnly (&&)()) = delete;
    Target(MoveOnly (&)()) = delete; // expected-note 2{{has been explicitly marked deleted here}}
    Target(MoveOnly (&&)());
};

CopyOnly make_copyonly();
MoveOnly make_moveonly();

Target t1() {
    CopyOnly (&r)() = make_copyonly;
    return r;
}

Target t2() {
    CopyOnly (&&r)() = static_cast<CopyOnly(&&)()>(make_copyonly);
    return r; // OK in all modes; not subject to implicit move
}

Target t3() {
    MoveOnly (&r)() = make_moveonly;
    return r; // expected-error {{invokes a deleted function}}
}

Target t4() {
    MoveOnly (&&r)() = static_cast<MoveOnly(&&)()>(make_moveonly);
    return r; // expected-error {{invokes a deleted function}}
}

} // namespace test_rvalue_ref_to_nonobject

// Both tests in test_constandnonconstcopy, and also test_conversion::test1, are
// "pure" C++98 tests (pretend 'delete' means 'private').
// However we may extend implicit moves into C++98, we must make sure the
// results in these are not changed.
namespace test_constandnonconstcopy {
struct ConstCopyOnly {
  ConstCopyOnly();
  ConstCopyOnly(ConstCopyOnly &) = delete; // cxx98-note {{marked deleted here}}
  ConstCopyOnly(const ConstCopyOnly &);
};
ConstCopyOnly t1() {
  ConstCopyOnly x;
  return x; // cxx98-error {{call to deleted constructor}}
}

struct NonConstCopyOnly {
  NonConstCopyOnly();
  NonConstCopyOnly(NonConstCopyOnly &);
  NonConstCopyOnly(const NonConstCopyOnly &) = delete; // cxx11_2b-note {{marked deleted here}}
};
NonConstCopyOnly t2() {
  NonConstCopyOnly x;
  return x; // cxx11_2b-error {{call to deleted constructor}}
}

} // namespace test_constandnonconstcopy

namespace test_conversion {

struct B;
struct A {
  A(B &) = delete; // cxx98-note {{has been explicitly deleted}}
};
struct B {
  operator A(); // cxx98-note {{candidate function}}
};
A test1(B x) { return x; } // cxx98-error-re {{conversion {{.*}} is ambiguous}}

struct C {};
struct D {
  operator C() &;
  operator C() const & = delete; // expected-note {{marked deleted here}}
};
C test2(D x) { return x; } // expected-error {{invokes a deleted function}}

} // namespace test_conversion

namespace test_simpler_implicit_move {

struct CopyOnly {
  CopyOnly(); // cxx2b-note {{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
  // cxx2b-note@-1 {{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
  CopyOnly(CopyOnly &); // cxx2b-note {{candidate constructor not viable: expects an lvalue for 1st argument}}
  // cxx2b-note@-1 {{candidate constructor not viable: expects an lvalue for 1st argument}}
};
struct MoveOnly {
  MoveOnly();
  MoveOnly(MoveOnly &&);
};
MoveOnly &&rref();

MoveOnly &&test1(MoveOnly &&w) {
  return w; // cxx98_20-error {{cannot bind to lvalue of type}}
}

CopyOnly test2(bool b) {
  static CopyOnly w1;
  CopyOnly w2;
  if (b) {
    return w1;
  } else {
    return w2; // cxx2b-error {{no matching constructor for initialization}}
  }
}

template <class T> T &&test3(T &&x) { return x; } // cxx98_20-error {{cannot bind to lvalue of type}}
template MoveOnly& test3<MoveOnly&>(MoveOnly&);
template MoveOnly &&test3<MoveOnly>(MoveOnly &&); // cxx98_20-note {{in instantiation of function template specialization}}

MoveOnly &&test4() {
  MoveOnly &&x = rref();
  return x; // cxx98_20-error {{cannot bind to lvalue of type}}
}

void test5() try {
  CopyOnly x;
  throw x; // cxx2b-error {{no matching constructor for initialization}}
} catch (...) {
}

} // namespace test_simpler_implicit_move

namespace test_auto_variables {

struct S {};

template <class T> struct range {
  S *begin() const;
  S *end() const;
};

template <class T> S test_dependent_ranged_for() {
  for (auto x : range<T>())
    return x;
  return S();
}
template S test_dependent_ranged_for<int>();

template <class T> struct X {};

template <class T> X<T> test_dependent_invalid_decl() {
  auto x = X<T>().foo(); // expected-error {{no member named 'foo'}}
  return x;
}
template X<int> test_dependent_invalid_decl<int>(); // expected-note {{requested here}}

} // namespace test_auto_variables
