// RUN: %clang_cc1 -std=c++20 -fsyntax-only -ast-dump -verify %s | FileCheck %s

// Ensure qualifiers are preserved during derived-to-base conversion. 
namespace PR127683 {

struct Base {
  int Val;
  mutable int MutVal;
  void f_const() const;
  void f_vol() volatile;
  void f_restrict() const __restrict;
};

struct Derived : Base { };
struct Derived1 : Base { };
struct Derived2 : Base { };
struct VeryDerived : Derived1, Derived2 { };
struct VDerived : virtual Base { };

// Value-initialize base class subobjects with type qualifiers.
Derived PlainObj{};
volatile Derived VObj;
const Derived CObj{}; // expected-note 2{{variable 'CObj' declared const here}}
const volatile Derived CVObj{}; // expected-note 2{{variable 'CVObj' declared const here}}
__attribute__((address_space(1))) Derived AddrSpaceObj{};

Derived *PlainPtr;
volatile Derived *VPtr;
const Derived *CPtr; // expected-note 2{{variable 'CPtr' declared const here}}
const volatile Derived *CVPtr; // expected-note 2{{variable 'CVPtr' declared const here}}
Derived *volatile VolPtr;
__attribute__((address_space(1))) Derived *ASPtr;
volatile __attribute__((address_space(1))) Derived *VASPtr;

volatile VeryDerived VVeryObj;
volatile VeryDerived *VVeryPtr;
volatile VDerived VVObj;
volatile VDerived *VVPtr;

void test_store() {
  // CHECK: `-ImplicitCastExpr {{.*}} 'volatile PR127683::Base' lvalue <UncheckedDerivedToBase (Base)>
  VObj.Val = 0;

  // CHECK: `-ImplicitCastExpr {{.*}} 'const PR127683::Base' lvalue <UncheckedDerivedToBase (Base)>
  CObj.Val = 1; // expected-error {{cannot assign to variable 'CObj' with const-qualified type 'const Derived'}}

  // CHECK: `-ImplicitCastExpr {{.*}} 'const volatile PR127683::Base' lvalue <UncheckedDerivedToBase (Base)>
  CVObj.Val = 1; // expected-error {{cannot assign to variable 'CVObj' with const-qualified type 'const volatile Derived'}}

  // CHECK: `-ImplicitCastExpr {{.*}} '__attribute__((address_space(1))) PR127683::Base' lvalue <UncheckedDerivedToBase (Base)>
  AddrSpaceObj.Val = 1;
}

void test_load() {
  // CHECK: `-ImplicitCastExpr {{.*}} <col:30> 'volatile PR127683::Base' lvalue <UncheckedDerivedToBase (Base)>
  [[maybe_unused]] int Val = VObj.Val;

  // CHECK: `-ImplicitCastExpr {{.*}} 'const PR127683::Base' lvalue <UncheckedDerivedToBase (Base)>
  Val = CObj.Val;
  
  // CHECK: `-ImplicitCastExpr {{.*}} 'const volatile PR127683::Base' lvalue <UncheckedDerivedToBase (Base)>
  Val = CVObj.Val;

  // CHECK: `-ImplicitCastExpr {{.*}} '__attribute__((address_space(1))) PR127683::Base' lvalue <UncheckedDerivedToBase (Base)>
  Val = AddrSpaceObj.Val;
}

void test_ptr_and_qualified() {
  // CHECK: `-ImplicitCastExpr {{.*}} 'volatile PR127683::Base *' <UncheckedDerivedToBase (Base)>
  VPtr->Val = 0;

  // CHECK: `-ImplicitCastExpr {{.*}} 'const PR127683::Base *' <UncheckedDerivedToBase (Base)>
  CPtr->Val = 1; // expected-error {{cannot assign to variable 'CPtr' with const-qualified type 'const Derived *'}}

  // CHECK: `-ImplicitCastExpr {{.*}} 'const volatile PR127683::Base *' <UncheckedDerivedToBase (Base)>
  CVPtr->Val = 1; // expected-error {{cannot assign to variable 'CVPtr' with const-qualified type 'const volatile Derived *'}}

  // CHECK: `-ImplicitCastExpr {{.*}} '__attribute__((address_space(1))) PR127683::Base *' <UncheckedDerivedToBase (Base)>
  ASPtr->Val = 1;

  // CHECK: `-ImplicitCastExpr {{.*}} 'volatile __attribute__((address_space(1))) PR127683::Base *' <UncheckedDerivedToBase (Base)>
  VASPtr->Val = 1;

  // Pointer itself is volatile, pointee is not.
  // CHECK: `-ImplicitCastExpr {{.*}} 'PR127683::Base *' <UncheckedDerivedToBase (Base)>
  VolPtr->Val = 1;

  // CHECK: `-ImplicitCastExpr {{.*}} 'Base *' <UncheckedDerivedToBase (Base)>
  VolPtr->Base::Val = 1;

  // Qualified member access on objects.
  // CHECK: `-ImplicitCastExpr {{.*}} 'volatile Base' lvalue <UncheckedDerivedToBase (Base)>
  VObj.Base::Val = 0;

  // CHECK: `-ImplicitCastExpr {{.*}} 'const Base' lvalue <UncheckedDerivedToBase (Base)>
  CObj.Base::Val = 1; // expected-error {{cannot assign to variable 'CObj' with const-qualified type 'const Derived'}}

  // CHECK: `-ImplicitCastExpr {{.*}} 'const volatile Base' lvalue <UncheckedDerivedToBase (Base)>
  CVObj.Base::Val = 1; // expected-error {{cannot assign to variable 'CVObj' with const-qualified type 'const volatile Derived'}}

  // CHECK: `-ImplicitCastExpr {{.*}} '__attribute__((address_space(1))) Base' lvalue <UncheckedDerivedToBase (Base)>
  AddrSpaceObj.Base::Val = 1;

  // Qualified member access on pointers.
  // CHECK: `-ImplicitCastExpr {{.*}} 'volatile Base *' <UncheckedDerivedToBase (Base)>
  VPtr->Base::Val = 0;

  // CHECK: `-ImplicitCastExpr {{.*}} 'const Base *' <UncheckedDerivedToBase (Base)>
  CPtr->Base::Val = 1; // expected-error {{cannot assign to variable 'CPtr' with const-qualified type 'const Derived *'}}

  // CHECK: `-ImplicitCastExpr {{.*}} 'const volatile Base *' <UncheckedDerivedToBase (Base)>
  CVPtr->Base::Val = 1; // expected-error {{cannot assign to variable 'CVPtr' with const-qualified type 'const volatile Derived *'}}

  // CHECK: `-ImplicitCastExpr {{.*}} '__attribute__((address_space(1))) Base *' <UncheckedDerivedToBase (Base)>
  ASPtr->Base::Val = 1;

  // CHECK: `-ImplicitCastExpr {{.*}} 'volatile __attribute__((address_space(1))) Base *' <UncheckedDerivedToBase (Base)>
  VASPtr->Base::Val = 1;
}

void test_diamond_and_virtual() {
  // CHECK: `-ImplicitCastExpr {{.*}} 'volatile PR127683::Base' lvalue <UncheckedDerivedToBase (Base)>
  // CHECK:   `-ImplicitCastExpr {{.*}} 'volatile Derived1' lvalue <UncheckedDerivedToBase (Derived1)>
  VVeryObj.Derived1::Val = 0;

  // CHECK: `-ImplicitCastExpr {{.*}} 'volatile PR127683::Base *' <UncheckedDerivedToBase (Base)>
  // CHECK:   `-ImplicitCastExpr {{.*}} 'volatile Derived1 *' <UncheckedDerivedToBase (Derived1)>
  VVeryPtr->Derived1::Val = 0;

  // CHECK: `-ImplicitCastExpr {{.*}} 'volatile PR127683::Base' lvalue <UncheckedDerivedToBase (virtual Base)>
  VVObj.Val = 0;

  // CHECK: `-ImplicitCastExpr {{.*}} 'volatile Base' lvalue <UncheckedDerivedToBase (virtual Base)>
  VVObj.Base::Val = 0;

  // CHECK: `-ImplicitCastExpr {{.*}} 'volatile PR127683::Base *' <UncheckedDerivedToBase (virtual Base)>
  VVPtr->Val = 0;

  // CHECK: `-ImplicitCastExpr {{.*}} 'volatile Base *' <UncheckedDerivedToBase (virtual Base)>
  VVPtr->Base::Val = 0;
}

void test_mutable_and_methods() {
  // CHECK: `-ImplicitCastExpr {{.*}} 'const volatile PR127683::Base' lvalue <UncheckedDerivedToBase (Base)>
  CVObj.MutVal = 1;

  // CHECK: `-ImplicitCastExpr {{.*}} 'const volatile Base' lvalue <UncheckedDerivedToBase (Base)>
  CVObj.Base::MutVal = 1;

  // CHECK: `-ImplicitCastExpr {{.*}} 'const volatile PR127683::Base *' <UncheckedDerivedToBase (Base)>
  CVPtr->MutVal = 1;

  // CHECK: `-ImplicitCastExpr {{.*}} 'const volatile Base *' <UncheckedDerivedToBase (Base)>
  CVPtr->Base::MutVal = 1;

  // Method calls preserve method pointer qualifiers (including __restrict).
  // CHECK: ImplicitCastExpr {{.*}} 'const PR127683::Base *' <UncheckedDerivedToBase (Base)>
  PlainPtr->f_const();
  // CHECK: ImplicitCastExpr {{.*}} 'Base *' <UncheckedDerivedToBase (Base)>
  PlainPtr->Base::f_const();
  // CHECK: ImplicitCastExpr {{.*}} 'volatile PR127683::Base *' <UncheckedDerivedToBase (Base)>
  VPtr->f_vol();
  // CHECK: ImplicitCastExpr {{.*}} 'volatile Base *' <UncheckedDerivedToBase (Base)>
  VPtr->Base::f_vol();
  // CHECK: ImplicitCastExpr {{.*}} 'const PR127683::Base *__restrict' <UncheckedDerivedToBase (Base)>
  PlainPtr->f_restrict();
  // CHECK: ImplicitCastExpr {{.*}} 'Base *' <UncheckedDerivedToBase (Base)>
  PlainPtr->Base::f_restrict();
}

using VolBase = volatile Base;
using ConstBase = const Base;

void test_qualified_typedef_nns() {
  // CHECK: `-ImplicitCastExpr {{.*}} 'Base' lvalue <UncheckedDerivedToBase (Base)>
  PlainObj.VolBase::Val = 1;

  // CHECK: `-ImplicitCastExpr {{.*}} 'Base' lvalue <UncheckedDerivedToBase (Base)>
  PlainObj.ConstBase::Val = 1;

  // CHECK: `-ImplicitCastExpr {{.*}} 'Base *' <UncheckedDerivedToBase (Base)>
  PlainPtr->VolBase::Val = 1;
}

}
