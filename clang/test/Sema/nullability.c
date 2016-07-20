// RUN: %clang_cc1 -fsyntax-only -fblocks -Wnullable-to-nonnull-conversion -Wno-nullability-declspec %s -verify

#if __has_feature(nullability)
#else
#  error nullability feature should be defined
#endif

typedef int * int_ptr;

// Parse nullability type specifiers.
// This note requires C11.
#if __STDC_VERSION__ > 199901L
// expected-note@+2{{'_Nonnull' specified here}}
#endif
typedef int * _Nonnull nonnull_int_ptr;
typedef int * _Nullable nullable_int_ptr;
typedef int * _Null_unspecified null_unspecified_int_ptr;

// Redundant nullability type specifiers.
typedef int * _Nonnull _Nonnull redundant_1; // expected-warning{{duplicate nullability specifier '_Nonnull'}}

// Conflicting nullability type specifiers.
typedef int * _Nonnull _Nullable conflicting_1; // expected-error{{nullability specifier '_Nonnull' conflicts with existing specifier '_Nullable'}}
typedef int * _Null_unspecified _Nonnull conflicting_2; // expected-error{{nullability specifier '_Null_unspecified' conflicts with existing specifier '_Nonnull'}}

// Redundant nullability specifiers via a typedef are okay.
typedef nonnull_int_ptr _Nonnull redundant_okay_1;

// Conflicting nullability specifiers via a typedef are not.
// Some of these errors require C11.
#if __STDC_VERSION__ > 199901L
typedef nonnull_int_ptr _Nullable conflicting_2; // expected-error{{nullability specifier '_Nullable' conflicts with existing specifier '_Nonnull'}}
#endif
typedef nonnull_int_ptr nonnull_int_ptr_typedef;
#if __STDC_VERSION__ > 199901L
typedef nonnull_int_ptr_typedef _Nullable conflicting_2; // expected-error{{nullability specifier '_Nullable' conflicts with existing specifier '_Nonnull'}}
#endif
typedef nonnull_int_ptr_typedef nonnull_int_ptr_typedef_typedef;
typedef nonnull_int_ptr_typedef_typedef _Null_unspecified conflicting_3; // expected-error{{nullability specifier '_Null_unspecified' conflicts with existing specifier '_Nonnull'}}

// Nullability applies to all pointer types.
typedef int (* _Nonnull function_pointer_type_1)(int, int);
typedef int (^ _Nonnull block_type_1)(int, int);

// Nullability must be on a pointer type.
typedef int _Nonnull int_type_1; // expected-error{{nullability specifier '_Nonnull' cannot be applied to non-pointer type 'int'}}

// Nullability can move out to a pointer/block pointer declarator
// (with a suppressed warning).
typedef _Nonnull int * nonnull_int_ptr_2;
typedef int _Nullable * nullable_int_ptr_2;
typedef _Nonnull int (* function_pointer_type_2)(int, int);
typedef _Nonnull int (^ block_type_2)(int, int);
typedef _Nonnull int * * _Nullable nonnull_int_ptr_ptr_1;
typedef _Nonnull int *(^ block_type_3)(int, int);
typedef _Nonnull int *(* function_pointer_type_3)(int, int);
typedef _Nonnull int_ptr (^ block_type_4)(int, int);
typedef _Nonnull int_ptr (* function_pointer_type_4)(int, int);
typedef void (* function_pointer_type_5)(int_ptr _Nonnull);

void acceptFunctionPtr(_Nonnull int *(*)(void));
void acceptBlockPtr(_Nonnull int *(^)(void));

void testBlockFunctionPtrNullability() {
  float *fp;
  fp = (function_pointer_type_3)0; // expected-warning{{from 'function_pointer_type_3' (aka 'int * _Nonnull (*)(int, int)')}}
  fp = (block_type_3)0; // expected-error{{from incompatible type 'block_type_3' (aka 'int * _Nonnull (^)(int, int)')}}
  fp = (function_pointer_type_4)0; // expected-warning{{from 'function_pointer_type_4' (aka 'int * _Nonnull (*)(int, int)')}}
  fp = (function_pointer_type_5)0; // expected-warning{{from 'function_pointer_type_5' (aka 'void (*)(int * _Nonnull)')}}
  fp = (block_type_4)0; // expected-error{{from incompatible type 'block_type_4' (aka 'int_ptr  _Nonnull (^)(int, int)')}}

  acceptFunctionPtr(0); // no-warning
  acceptBlockPtr(0); // no-warning
}

// Moving nullability where it creates a conflict.
typedef _Nonnull int * _Nullable *  conflict_int_ptr_ptr_2; // expected-error{{nullability specifier '_Nonnull' cannot be applied to non-pointer type 'int'}}

// Nullability is not part of the canonical type.
typedef int * _Nonnull ambiguous_int_ptr;
// Redefining a typedef is a C11 feature.
#if __STDC_VERSION__ > 199901L
typedef int * ambiguous_int_ptr;
typedef int * _Nullable ambiguous_int_ptr;
#endif

// Printing of nullability.
float f;
int * _Nonnull ip_1 = &f; // expected-warning{{incompatible pointer types initializing 'int * _Nonnull' with an expression of type 'float *'}}

// Check printing of nullability specifiers.
void printing_nullability(void) {
  int * _Nonnull iptr;
  float *fptr = iptr; // expected-warning{{incompatible pointer types initializing 'float *' with an expression of type 'int * _Nonnull'}}

  int * * _Nonnull iptrptr;
  float **fptrptr = iptrptr; // expected-warning{{incompatible pointer types initializing 'float **' with an expression of type 'int ** _Nonnull'}}

  int * _Nullable * _Nonnull iptrptr2;
  float * *fptrptr2 = iptrptr2; // expected-warning{{incompatible pointer types initializing 'float **' with an expression of type 'int * _Nullable * _Nonnull'}}
}

// Check passing null to a _Nonnull argument.
void accepts_nonnull_1(_Nonnull int *ptr);
void (*accepts_nonnull_2)(_Nonnull int *ptr);
void (^accepts_nonnull_3)(_Nonnull int *ptr);

void test_accepts_nonnull_null_pointer_literal() {
  accepts_nonnull_1(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
  accepts_nonnull_2(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
  accepts_nonnull_3(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
}

// Check returning nil from a _Nonnull-returning function.
_Nonnull int *returns_int_ptr(int x) {
  if (x) {
    return 0; // expected-warning{{null returned from function that requires a non-null return value}}
  }

  return (_Nonnull int *)0;
}

// Check nullable-to-nonnull conversions.
void nullable_to_nonnull(_Nullable int *ptr) {
  int *a = ptr; // okay
  _Nonnull int *b = ptr; // expected-warning{{implicit conversion from nullable pointer 'int * _Nullable' to non-nullable pointer type 'int * _Nonnull'}}
  b = ptr; // expected-warning{{implicit conversion from nullable pointer 'int * _Nullable' to non-nullable pointer type 'int * _Nonnull'}}

  accepts_nonnull_1(ptr); // expected-warning{{implicit conversion from nullable pointer 'int * _Nullable' to non-nullable pointer type 'int * _Nonnull'}}
}

// Check nullability of conditional expressions.
void conditional_expr(int c) {
  int * _Nonnull p;
  int * _Nonnull nonnullP;
  int * _Nullable nullableP;
  int * _Null_unspecified unspecifiedP;
  int *noneP;

  p = c ? nonnullP : nonnullP;
  p = c ? nonnullP : nullableP; // expected-warning{{implicit conversion from nullable pointer 'int * _Nullable' to non-nullable pointer type 'int * _Nonnull'}}
  p = c ? nonnullP : unspecifiedP;
  p = c ? nonnullP : noneP;
  p = c ? nullableP : nonnullP; // expected-warning{{implicit conversion from nullable pointer 'int * _Nullable' to non-nullable pointer type 'int * _Nonnull'}}
  p = c ? nullableP : nullableP; // expected-warning{{implicit conversion from nullable pointer 'int * _Nullable' to non-nullable pointer type 'int * _Nonnull'}}
  p = c ? nullableP : unspecifiedP; // expected-warning{{implicit conversion from nullable pointer 'int * _Nullable' to non-nullable pointer type 'int * _Nonnull'}}
  p = c ? nullableP : noneP; // expected-warning{{implicit conversion from nullable pointer 'int * _Nullable' to non-nullable pointer type 'int * _Nonnull'}}
  p = c ? unspecifiedP : nonnullP;
  p = c ? unspecifiedP : nullableP; // expected-warning{{implicit conversion from nullable pointer 'int * _Nullable' to non-nullable pointer type 'int * _Nonnull'}}
  p = c ? unspecifiedP : unspecifiedP;
  p = c ? unspecifiedP : noneP;
  p = c ? noneP : nonnullP;
  p = c ? noneP : nullableP; // expected-warning{{implicit conversion from nullable pointer 'int * _Nullable' to non-nullable pointer type 'int * _Nonnull'}}
  p = c ? noneP : unspecifiedP;
  p = c ? noneP : noneP;

  // Check that we don't remove all sugar when creating a new QualType for the
  // conditional expression.
  typedef int *IntP;
  typedef IntP _Nonnull NonnullIntP0;
  typedef NonnullIntP0 _Nonnull NonnullIntP1;
  typedef IntP _Nullable NullableIntP0;
  typedef NullableIntP0 _Nullable NullableIntP1;
  NonnullIntP1 nonnullP2;
  NullableIntP1 nullableP2;

  p = c ? nonnullP2 : nonnullP2;
  p = c ? nonnullP2 : nullableP2; // expected-warning{{implicit conversion from nullable pointer 'IntP _Nullable' (aka 'int *') to non-nullable pointer type 'int * _Nonnull'}}
  p = c ? nullableP2 : nonnullP2; // expected-warning{{implicit conversion from nullable pointer 'NullableIntP1' (aka 'int *') to non-nullable pointer type 'int * _Nonnull'}}
  p = c ? nullableP2 : nullableP2; // expected-warning{{implicit conversion from nullable pointer 'NullableIntP1' (aka 'int *') to non-nullable pointer type 'int * _Nonnull'}}
}

// Check nullability of binary conditional expressions.
void binary_conditional_expr() {
  int * _Nonnull p;
  int * _Nonnull nonnullP;
  int * _Nullable nullableP;
  int * _Null_unspecified unspecifiedP;
  int *noneP;

  p = nonnullP ?: nonnullP;
  p = nonnullP ?: nullableP;
  p = nonnullP ?: unspecifiedP;
  p = nonnullP ?: noneP;
  p = nullableP ?: nonnullP;
  p = nullableP ?: nullableP; // expected-warning{{implicit conversion from nullable pointer 'int * _Nullable' to non-nullable pointer type 'int * _Nonnull'}}
  p = nullableP ?: unspecifiedP;
  p = nullableP ?: noneP;
  p = unspecifiedP ?: nonnullP;
  p = unspecifiedP ?: nullableP; // expected-warning{{implicit conversion from nullable pointer 'int * _Nullable' to non-nullable pointer type 'int * _Nonnull'}}
  p = unspecifiedP ?: unspecifiedP;
  p = unspecifiedP ?: noneP;
  p = noneP ?: nonnullP;
  p = noneP ?: nullableP; // expected-warning{{implicit conversion from nullable pointer 'int * _Nullable' to non-nullable pointer type 'int * _Nonnull'}}
  p = noneP ?: unspecifiedP;
  p = noneP ?: noneP;
}
