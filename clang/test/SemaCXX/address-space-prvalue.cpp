// RUN: %clang_cc1 %s -fsyntax-only -verify

struct S {          // #ctors
  S(int);           // #sint
  void f();
  void cf() const;
};

using as1_S = __attribute__((address_space(1))) S;
using const_as1_S = const __attribute__((address_space(1))) S;

as1_S make();
as1_S &getRef();
const_as1_S makeConst();

void test() {
  // S has no constructor that can build an object in address space 1.
  as1_S{1};
  // expected-error@-1 {{no matching constructor for initialization of 'as1_S' (aka '__attribute__((address_space(1))) S')}}
  // expected-note@#ctors 2 {{candidate constructor ignored: cannot be used to construct an object in address space '__attribute__((address_space(1)))'}}
  // expected-note@#sint {{candidate constructor ignored: cannot be used to construct an object in address space '__attribute__((address_space(1)))'}}

  // The function call is a prvalue, so the address space is stripped from its
  // type. The temporary materialized for the member call is a plain 'S', so the
  // call is well-formed.
  make().f();

  // The prvalue drops the address space but retains const. The materialized
  // temporary is 'const S', so the const member call binds.
  makeConst().cf();
  
  // The reference is not a prvalue, so the address space is retained.
  // The member call cannot bind 'this' to an object in address space 1.
  getRef().f();
  // expected-error@-1 {{cannot initialize object parameter of type 'S' with an expression of type 'as1_S' (aka '__attribute__((address_space(1))) S')}}

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
