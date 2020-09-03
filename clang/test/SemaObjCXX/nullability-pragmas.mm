// RUN: %clang_cc1 -fsyntax-only -fblocks -I %S/Inputs %s -verify

#include "nullability-pragmas-1.h"
#include "nullability-pragmas-2.h"
#include "nullability-pragmas-generics-1.h"

#if !__has_feature(assume_nonnull)
#  error assume_nonnull feature is not set
#endif

#if !__has_extension(assume_nonnull)
#  error assume_nonnull extension is not set
#endif

void test_pragmas_1(A * _Nonnull a, AA * _Nonnull aa) {
  f1(0); // okay: no nullability annotations
  f2(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
  f3(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
  f4(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
  f5(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
  f6(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
  f7(0); // okay
  f8(0); // okay
  f9(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
  f10(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
  f11(0); // okay
  f12(0); // okay
  [a method1:0]; // expected-warning{{null passed to a callee that requires a non-null argument}}

  f17(a); // expected-error{{no matching function for call to 'f17'}}
  [a method3: a]; // expected-error{{cannot initialize a parameter of type 'NSError * _Nullable * _Nullable' with an lvalue of type 'A * _Nonnull'}}
  [a method4: a]; // expected-error{{cannot initialize a parameter of type 'NSErrorPtr  _Nullable * _Nullable' (aka 'NSError **') with an lvalue of type 'A * _Nonnull'}}

  float *ptr;
  ptr = f13(); // expected-error{{incompatible pointer types assigning to 'float *' from 'int_ptr _Nonnull' (aka 'int *')}}
  ptr = f14(); // expected-error{{incompatible pointer types assigning to 'float *' from 'A * _Nonnull'}}
  ptr = [a method1:a]; // expected-error{{incompatible pointer types assigning to 'float *' from 'A * _Nonnull'}}
  ptr = a.aProp; // expected-error{{incompatible pointer types assigning to 'float *' from 'A * _Nonnull'}}
  ptr = global_int_ptr; // expected-error{{incompatible pointer types assigning to 'float *' from 'int * _Nonnull'}}
  ptr = f15(); // expected-error{{incompatible pointer types assigning to 'float *' from 'int * _Null_unspecified'}}
  ptr = f16(); // expected-error{{incompatible pointer types assigning to 'float *' from 'A * _Null_unspecified'}}
  ptr = [a method2]; // expected-error{{incompatible pointer types assigning to 'float *' from 'A * _Null_unspecified'}}

  ptr = aa->ivar1; // expected-error{{incompatible pointer types assigning to 'float *' from 'id'}}
  ptr = aa->ivar2; // expected-error{{incompatible pointer types assigning to 'float *' from 'id _Nonnull'}}
}

void test_pragmas_generics(void) {
  float *fp;

  NSGeneric<C *> *genC;
  fp = [genC tee]; // expected-error{{incompatible pointer types assigning to 'float *' from 'C *'}}
  fp = [genC maybeTee]; // expected-error{{incompatible pointer types assigning to 'float *' from 'C * _Nullable'}}

  Generic_with_C genC2;
  fp = genC2; // expected-error{{incompatible pointer types assigning to 'float *' from 'Generic_with_C' (aka 'NSGeneric<C *> *')}}
}
