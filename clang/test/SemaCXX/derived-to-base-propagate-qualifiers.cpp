// RUN: %clang_cc1 -std=c++20 -fsyntax-only -ast-dump -verify %s | FileCheck %s

// Ensure qualifiers are preserved during derived-to-base conversion. 
namespace PR127683 {

struct Base {
  int Val;
};

struct Derived : Base { };

// Value-initialize base class subobjects with type qualifiers.
volatile Derived VObj;
const Derived CObj{}; // expected-note{{variable 'CObj' declared const here}}
const volatile Derived CVObj{}; // expected-note{{variable 'CVObj' declared const here}}
__attribute__((address_space(1))) Derived AddrSpaceObj{};

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

}
