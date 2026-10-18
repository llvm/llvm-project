// RUN: %clang_cc1 %s -fsyntax-only -verify

struct S {                                     // #ctors
  S(int);                                      // #sint
  void f() const;                              // #f-const
  void f() __attribute__((address_space(1)));  // #f-as \
                                               // expected-error {{function type may not be qualified with an address space}}
};

using const_int = const int;
using as1_int = __attribute__((address_space(1))) int;
using const_S = const S;
using as1_S = __attribute__((address_space(1))) S;

void testQualifiers() {
  // Ok; const is dropped on prvalues of non-class type.
  (void)(const_int{1});
  // Ok; address space is dropped on prvalues of non-class type.
  (void)(as1_int{1});
  // Ok; const is retained on prvalues of class type; const qualified
  // member function called.
  const_S{1}.f();
  // Error; address space is retained on prvalues of class type, but no
  // constructor or member function can be called.
  as1_S{1}.f();
  // expected-error@-1 {{no matching constructor for initialization of 'as1_S' (aka '__attribute__((address_space(1))) S')}}
  // expected-error@-2 {{no matching member function for call to 'f'}}
  // expected-note@#ctors 2 {{candidate constructor ignored: cannot be used to construct an object in address space '__attribute__((address_space(1)))'}}
  // expected-note@#sint {{candidate constructor ignored: cannot be used to construct an object in address space '__attribute__((address_space(1)))'}}
  // expected-note@#f-const {{candidate function not viable: 'this' object is in address space '1', but method expects object in generic address space}}
  // expected-note@#f-as {{candidate function not viable: 'this' object is in address space '1', but method expects object in generic address space}}
}

void temporaryMaterializationTest() {
  // An address-space-qualified class temporary retains the address space on
  // the materialized object, so no constructor can be used.
  as1_S{0};
  // expected-error@-1 {{no matching constructor for initialization of 'as1_S' (aka '__attribute__((address_space(1))) S')}}
  // expected-note@#ctors 2 {{candidate constructor ignored: cannot be used to construct an object in address space '__attribute__((address_space(1)))'}}
  // expected-note@#sint {{candidate constructor ignored: cannot be used to construct an object in address space '__attribute__((address_space(1)))'}}
}

// FIXME: All qualifiers including address space are retained on array elements
//        The code in getNonLValueExprType() to remove qualifiers from prvalues
//        acts on the array type and not the element. The code to remove address
//        spaces is never hit. I am not sure this if this is correct behavior.
using int_array = const __attribute__((address_space(1))) int[1];
using S_array = const __attribute__((address_space(1))) S[1];
void arrayTest() {
  // Passes because scalar elements can be initialized directly.
  (void)(int_array{1});
  // Errors because class elements cannot be constructed in the address space.
  (void)(S_array{1});
  // expected-error@-1 {{no viable conversion from 'int' to 'const __attribute__((address_space(1))) S'}}
  // expected-note@#ctors {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int' to 'const S &' for 1st argument}}
  // expected-note@#ctors {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'int' to 'S &&' for 1st argument}}
  // expected-note@#sint {{candidate constructor ignored: cannot be used to construct an object in address space '__attribute__((address_space(1)))'}}
}

