// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fcxx-exceptions -verify=expected,cxx20 %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -fcxx-exceptions -verify=expected,cxx11_14_17 %s
// RUN: %clang_cc1 -std=c++14 -fsyntax-only -fcxx-exceptions -verify=expected,cxx11_14_17 %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fcxx-exceptions -verify=expected,cxx11_14_17 %s

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

// In C++20, implicitly movable entity can be rvalue reference to non-volatile
// automatic object.
namespace test_implicitly_movable_rvalue_ref {
struct A1 {
  A1(A1 &&);
  A1(const A1 &) = delete; // cxx11_14_17-note {{'A1' has been explicitly marked deleted here}}
};
A1 test1(A1 &&a) {
  return a; // cxx11_14_17-error {{call to deleted constructor of 'test_implicitly_movable_rvalue_ref::A1'}}
}

struct A2 {
  A2(A2 &&);

private:
  A2(const A2 &); // cxx11_14_17-note {{declared private here}}
};
A2 test2(A2 &&a) {
  return a; // cxx11_14_17-error {{calling a private constructor of class 'test_implicitly_movable_rvalue_ref::A2'}}
}

struct B1 {
  B1(const B1 &);
  B1(B1 &&) = delete; // cxx20-note {{'B1' has been explicitly marked deleted here}}
};
B1 test3(B1 &&b) {
  return b; // cxx20-error {{call to deleted constructor of 'test_implicitly_movable_rvalue_ref::B1'}}
}

struct B2 {
  B2(const B2 &);

private:
  B2(B2 &&); // cxx20-note {{declared private here}}
};
B2 test4(B2 &&b) {
  return b; // cxx20-error {{calling a private constructor of class 'test_implicitly_movable_rvalue_ref::B2'}}
}
} // namespace test_implicitly_movable_rvalue_ref

// In C++20, operand of throw-expression can be function parameter or
// catch-clause parameter.
namespace test_throw_parameter {
void func();

struct A1 {
  A1(const A1 &);
  A1(A1 &&) = delete; // cxx20-note {{'A1' has been explicitly marked deleted here}}
};
void test1() {
  try {
    func();
  } catch (A1 a) {
    throw a; // cxx20-error {{call to deleted constructor of 'test_throw_parameter::A1'}}
  }
}

struct A2 {
  A2(const A2 &);

private:
  A2(A2 &&); // cxx20-note {{declared private here}}
};
void test2() {
  try {
    func();
  } catch (A2 a) {
    throw a; // cxx20-error {{calling a private constructor of class 'test_throw_parameter::A2'}}
  }
}
} // namespace test_throw_parameter

// In C++20, during the first overload resolution, the selected function no
// need to be a constructor.
namespace test_non_ctor_conversion {
class C {};

struct A1 {
  operator C() &&;
  operator C() const & = delete; // cxx11_14_17-note {{'operator C' has been explicitly marked deleted here}}
};
C test1() {
  A1 a;
  return a; // cxx11_14_17-error {{conversion function from 'test_non_ctor_conversion::A1' to 'test_non_ctor_conversion::C' invokes a deleted function}}
}

struct A2 {
  operator C() &&;

private:
  operator C() const &; // cxx11_14_17-note {{declared private here}}
};
C test2() {
  A2 a;
  return a; // cxx11_14_17-error {{'operator C' is a private member of 'test_non_ctor_conversion::A2'}}
}

struct B1 {
  operator C() const &;
  operator C() && = delete; // cxx20-note {{'operator C' has been explicitly marked deleted here}}
};
C test3() {
  B1 b;
  return b; // cxx20-error {{conversion function from 'test_non_ctor_conversion::B1' to 'test_non_ctor_conversion::C' invokes a deleted function}}
}

struct B2 {
  operator C() const &;

private:
  operator C() &&; // cxx20-note {{declared private here}}
};
C test4() {
  B2 b;
  return b; // cxx20-error {{'operator C' is a private member of 'test_non_ctor_conversion::B2'}}
}
} // namespace test_non_ctor_conversion

// In C++20, during the first overload resolution, the first parameter of the
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
  NeedValue(A1); // cxx11_14_17-note 2 {{passing argument to parameter here}}
  NeedValue(A2);
  NeedValue(B1); // cxx20-note 2 {{passing argument to parameter here}}
  NeedValue(B2);
};

struct A1 {
  A1();
  A1(A1 &&);
  A1(const A1 &) = delete; // cxx11_14_17-note 3 {{'A1' has been explicitly marked deleted here}}
};
NeedValue test_1_1() {
  // not rvalue reference
  // same type
  A1 a;
  return a; // cxx11_14_17-error {{call to deleted constructor of 'test_ctor_param_rvalue_ref::A1'}}
}
class DerivedA1 : public A1 {};
A1 test_1_2() {
  // rvalue reference
  // not same type
  DerivedA1 a;
  return a; // cxx11_14_17-error {{call to deleted constructor of 'test_ctor_param_rvalue_ref::A1'}}
}
NeedValue test_1_3() {
  // not rvalue reference
  // not same type
  DerivedA1 a;
  return a; // cxx11_14_17-error {{call to deleted constructor of 'test_ctor_param_rvalue_ref::A1'}}
}

struct A2 {
  A2();
  A2(A2 &&);

private:
  A2(const A2 &); // cxx11_14_17-note 3 {{declared private here}}
};
NeedValue test_2_1() {
  // not rvalue reference
  // same type
  A2 a;
  return a; // cxx11_14_17-error {{calling a private constructor of class 'test_ctor_param_rvalue_ref::A2'}}
}
class DerivedA2 : public A2 {};
A2 test_2_2() {
  // rvalue reference
  // not same type
  DerivedA2 a;
  return a; // cxx11_14_17-error {{calling a private constructor of class 'test_ctor_param_rvalue_ref::A2'}}
}
NeedValue test_2_3() {
  // not rvalue reference
  // not same type
  DerivedA2 a;
  return a; // cxx11_14_17-error {{calling a private constructor of class 'test_ctor_param_rvalue_ref::A2'}}
}

struct B1 {
  B1();
  B1(const B1 &);
  B1(B1 &&) = delete; // cxx20-note 3 {{'B1' has been explicitly marked deleted here}}
};
NeedValue test_3_1() {
  // not rvalue reference
  // same type
  B1 b;
  return b; // cxx20-error {{call to deleted constructor of 'test_ctor_param_rvalue_ref::B1'}}
}
class DerivedB1 : public B1 {};
B1 test_3_2() {
  // rvalue reference
  // not same type
  DerivedB1 b;
  return b; // cxx20-error {{call to deleted constructor of 'test_ctor_param_rvalue_ref::B1'}}
}
NeedValue test_3_3() {
  // not rvalue reference
  // not same type
  DerivedB1 b;
  return b; // cxx20-error {{call to deleted constructor of 'test_ctor_param_rvalue_ref::B1'}}
}

struct B2 {
  B2();
  B2(const B2 &);

private:
  B2(B2 &&); // cxx20-note 3 {{declared private here}}
};
NeedValue test_4_1() {
  // not rvalue reference
  // same type
  B2 b;
  return b; // cxx20-error {{calling a private constructor of class 'test_ctor_param_rvalue_ref::B2'}}
}
class DerivedB2 : public B2 {};
B2 test_4_2() {
  // rvalue reference
  // not same type
  DerivedB2 b;
  return b; // cxx20-error {{calling a private constructor of class 'test_ctor_param_rvalue_ref::B2'}}
}
NeedValue test_4_3() {
  // not rvalue reference
  // not same type
  DerivedB2 b;
  return b; // cxx20-error {{calling a private constructor of class 'test_ctor_param_rvalue_ref::B2'}}
}
} // namespace test_ctor_param_rvalue_ref

namespace test_lvalue_ref_is_not_moved_from {

struct Target {};
  // expected-note@-1 {{candidate constructor (the implicit copy constructor) not viable}}
  // expected-note@-2 {{candidate constructor (the implicit move constructor) not viable}}
  // cxx11_14_17-note@-3 {{candidate constructor (the implicit copy constructor) not viable}}
  // cxx11_14_17-note@-4 {{candidate constructor (the implicit move constructor) not viable}}

struct CopyOnly {
  CopyOnly(CopyOnly&&) = delete; // cxx20-note {{has been explicitly marked deleted here}}
  CopyOnly(CopyOnly&);
  operator Target() && = delete; // cxx20-note {{has been explicitly marked deleted here}}
  operator Target() &;
};

struct MoveOnly {
  MoveOnly(MoveOnly&&); // expected-note {{copy constructor is implicitly deleted because}}
    // cxx11_14_17-note@-1 {{copy constructor is implicitly deleted because}}
  operator Target() &&; // expected-note {{candidate function not viable}}
    // cxx11_14_17-note@-1 {{candidate function not viable}}
};

extern CopyOnly copyonly;
extern MoveOnly moveonly;

CopyOnly t1() {
    CopyOnly& r = copyonly;
    return r;
}

CopyOnly t2() {
    CopyOnly&& r = static_cast<CopyOnly&&>(copyonly);
    return r; // cxx20-error {{call to deleted constructor}}
}

MoveOnly t3() {
    MoveOnly& r = moveonly;
    return r; // expected-error {{call to implicitly-deleted copy constructor}}
}

MoveOnly t4() {
    MoveOnly&& r = static_cast<MoveOnly&&>(moveonly);
    return r; // cxx11_14_17-error {{call to implicitly-deleted copy constructor}}
}

Target t5() {
    CopyOnly& r = copyonly;
    return r;
}

Target t6() {
    CopyOnly&& r = static_cast<CopyOnly&&>(copyonly);
    return r; // cxx20-error {{invokes a deleted function}}
}

Target t7() {
    MoveOnly& r = moveonly;
    return r; // expected-error {{no viable conversion}}
}

Target t8() {
    MoveOnly&& r = static_cast<MoveOnly&&>(moveonly);
    return r; // cxx11_14_17-error {{no viable conversion}}
}

} // namespace test_lvalue_ref_is_not_moved_from

namespace test_rvalue_ref_to_nonobject {

struct CopyOnly {};
struct MoveOnly {};

struct Target {
    Target(CopyOnly (&)());
    Target(CopyOnly (&&)()) = delete;
    Target(MoveOnly (&)()) = delete; // expected-note {{has been explicitly marked deleted here}}
      // expected-note@-1 {{has been explicitly marked deleted here}}
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
