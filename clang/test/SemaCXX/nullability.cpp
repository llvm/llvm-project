// RUN: %clang_cc1 -std=c++11 -fsyntax-only -Wno-nullability-declspec %s -verify -Wnullable-to-nonnull-conversion

#if __has_feature(nullability)
#else
#  error nullability feature should be defined
#endif

typedef decltype(nullptr) nullptr_t;

class X {
};

// Nullability applies to all pointer types.
typedef int (X::* _Nonnull member_function_type_1)(int);
typedef int X::* _Nonnull member_data_type_1;
typedef nullptr_t _Nonnull nonnull_nullptr_t; // expected-error{{nullability specifier '_Nonnull' cannot be applied to non-pointer type 'nullptr_t'}}

// Nullability can move into member pointers (this is suppressing a warning).
typedef _Nonnull int (X::* member_function_type_2)(int);
typedef int (X::* _Nonnull member_function_type_3)(int);
typedef _Nonnull int X::* member_data_type_2;

// Adding non-null via a template.
template<typename T>
struct AddNonNull {
  typedef _Nonnull T type; // expected-error{{nullability specifier '_Nonnull' cannot be applied to non-pointer type 'int'}}
  // expected-error@-1{{nullability specifier '_Nonnull' cannot be applied to non-pointer type 'nullptr_t'}}
};

typedef AddNonNull<int *>::type nonnull_int_ptr_1;
typedef AddNonNull<int * _Nullable>::type nonnull_int_ptr_2; // FIXME: check that it was overridden
typedef AddNonNull<nullptr_t>::type nonnull_int_ptr_3; // expected-note{{in instantiation of template class}}

typedef AddNonNull<int>::type nonnull_non_pointer_1; // expected-note{{in instantiation of template class 'AddNonNull<int>' requested here}}

// Non-null checking within a template.
template<typename T>
struct AddNonNull2 {
  typedef _Nonnull AddNonNull<T> invalid1; // expected-error{{nullability specifier '_Nonnull' cannot be applied to non-pointer type 'AddNonNull<T>'}}
  typedef _Nonnull AddNonNull2 invalid2; // expected-error{{nullability specifier '_Nonnull' cannot be applied to non-pointer type 'AddNonNull2<T>'}}
  typedef _Nonnull AddNonNull2<T> invalid3; // expected-error{{nullability specifier '_Nonnull' cannot be applied to non-pointer type 'AddNonNull2<T>'}}
  typedef _Nonnull typename AddNonNull<T>::type okay1;

  // Don't move past a dependent type even if we know that nullability
  // cannot apply to that specific dependent type.
  typedef _Nonnull AddNonNull<T> (*invalid4); // expected-error{{nullability specifier '_Nonnull' cannot be applied to non-pointer type 'AddNonNull<T>'}}
};

// Check passing null to a _Nonnull argument.
void (*accepts_nonnull_1)(_Nonnull int *ptr);
void (*& accepts_nonnull_2)(_Nonnull int *ptr) = accepts_nonnull_1;
void (X::* accepts_nonnull_3)(_Nonnull int *ptr);
void accepts_nonnull_4(_Nonnull int *ptr);
void (&accepts_nonnull_5)(_Nonnull int *ptr) = accepts_nonnull_4;

void test_accepts_nonnull_null_pointer_literal(X *x) {
  accepts_nonnull_1(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
  accepts_nonnull_2(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
  (x->*accepts_nonnull_3)(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
  accepts_nonnull_4(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
  accepts_nonnull_5(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
}

template<void FP(_Nonnull int*)> 
void test_accepts_nonnull_null_pointer_literal_template() {
  FP(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
}

template void test_accepts_nonnull_null_pointer_literal_template<&accepts_nonnull_4>(); // expected-note{{instantiation of function template specialization}}

void TakeNonnull(void *_Nonnull);
// Check different forms of assignment to a nonull type from a nullable one.
void AssignAndInitNonNull() {
  void *_Nullable nullable;
  void *_Nonnull p(nullable); // expected-warning{{implicit conversion from nullable pointer 'void * _Nullable' to non-nullable pointer type 'void * _Nonnull'}}
  void *_Nonnull p2{nullable}; // expected-warning{{implicit conversion from nullable pointer 'void * _Nullable' to non-nullable pointer type 'void * _Nonnull'}}
  void *_Nonnull p3 = {nullable}; // expected-warning{{implicit conversion from nullable pointer 'void * _Nullable' to non-nullable pointer type 'void * _Nonnull'}}
  void *_Nonnull p4 = nullable; // expected-warning{{implicit conversion from nullable pointer 'void * _Nullable' to non-nullable pointer type 'void * _Nonnull'}}
  void *_Nonnull nonnull;
  nonnull = nullable; // expected-warning{{implicit conversion from nullable pointer 'void * _Nullable' to non-nullable pointer type 'void * _Nonnull'}}
  nonnull = {nullable}; // expected-warning{{implicit conversion from nullable pointer 'void * _Nullable' to non-nullable pointer type 'void * _Nonnull'}}

  TakeNonnull(nullable); //expected-warning{{implicit conversion from nullable pointer 'void * _Nullable' to non-nullable pointer type 'void * _Nonnull}}
  TakeNonnull(nonnull); // OK
}

void *_Nullable ReturnNullable();

void AssignAndInitNonNullFromFn() {
  void *_Nonnull p(ReturnNullable()); // expected-warning{{implicit conversion from nullable pointer 'void * _Nullable' to non-nullable pointer type 'void * _Nonnull'}}
  void *_Nonnull p2{ReturnNullable()}; // expected-warning{{implicit conversion from nullable pointer 'void * _Nullable' to non-nullable pointer type 'void * _Nonnull'}}
  void *_Nonnull p3 = {ReturnNullable()}; // expected-warning{{implicit conversion from nullable pointer 'void * _Nullable' to non-nullable pointer type 'void * _Nonnull'}}
  void *_Nonnull p4 = ReturnNullable(); // expected-warning{{implicit conversion from nullable pointer 'void * _Nullable' to non-nullable pointer type 'void * _Nonnull'}}
  void *_Nonnull nonnull;
  nonnull = ReturnNullable(); // expected-warning{{implicit conversion from nullable pointer 'void * _Nullable' to non-nullable pointer type 'void * _Nonnull'}}
  nonnull = {ReturnNullable()}; // expected-warning{{implicit conversion from nullable pointer 'void * _Nullable' to non-nullable pointer type 'void * _Nonnull'}}

  TakeNonnull(ReturnNullable()); //expected-warning{{implicit conversion from nullable pointer 'void * _Nullable' to non-nullable pointer type 'void * _Nonnull}}
}

void ConditionalExpr(bool c) {
  struct Base {};
  struct Derived : Base {};

  Base * _Nonnull p;
  Base * _Nonnull nonnullB;
  Base * _Nullable nullableB;
  Derived * _Nonnull nonnullD;
  Derived * _Nullable nullableD;

  p = c ? nonnullB : nonnullD;
  p = c ? nonnullB : nullableD; // expected-warning{{implicit conversion from nullable pointer 'Base * _Nullable' to non-nullable pointer type 'Base * _Nonnull}}
  p = c ? nullableB : nonnullD; // expected-warning{{implicit conversion from nullable pointer 'Base * _Nullable' to non-nullable pointer type 'Base * _Nonnull}}
  p = c ? nullableB : nullableD; // expected-warning{{implicit conversion from nullable pointer 'Base * _Nullable' to non-nullable pointer type 'Base * _Nonnull}}
  p = c ? nonnullD : nonnullB;
  p = c ? nonnullD : nullableB; // expected-warning{{implicit conversion from nullable pointer 'Base * _Nullable' to non-nullable pointer type 'Base * _Nonnull}}
  p = c ? nullableD : nonnullB; // expected-warning{{implicit conversion from nullable pointer 'Base * _Nullable' to non-nullable pointer type 'Base * _Nonnull}}
  p = c ? nullableD : nullableB; // expected-warning{{implicit conversion from nullable pointer 'Base * _Nullable' to non-nullable pointer type 'Base * _Nonnull}}
}

void arraysInLambdas() {
  typedef int INTS[4];
  auto simple = [](int [_Nonnull 2]) {};
  simple(nullptr); // expected-warning {{null passed to a callee that requires a non-null argument}}
  auto nested = [](void *_Nullable [_Nonnull 2]) {};
  nested(nullptr); // expected-warning {{null passed to a callee that requires a non-null argument}}
  auto nestedBad = [](int [2][_Nonnull 2]) {}; // expected-error {{nullability specifier '_Nonnull' cannot be applied to non-pointer type 'int [2]'}}

  auto withTypedef = [](INTS _Nonnull) {};
  withTypedef(nullptr); // expected-warning {{null passed to a callee that requires a non-null argument}}
  auto withTypedefBad = [](INTS _Nonnull[2]) {}; // expected-error {{nullability specifier '_Nonnull' cannot be applied to non-pointer type 'INTS' (aka 'int [4]')}}
}
