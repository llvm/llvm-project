// RUN: %clang_cc1 -std=c++11 -fsyntax-only -Wno-nullability-declspec %s -verify -Wnullable-to-nonnull-conversion -I%S/Inputs

#if __has_feature(nullability)
#else
#  error nullability feature should be defined
#endif
#if __has_feature(nullability_on_classes)
#else
#  error smart-pointer feature should be defined
#endif

#include "nullability-completeness.h"

typedef decltype(nullptr) nullptr_t;

class X {
};

// Nullability applies to all pointer types.
typedef int (X::* _Nonnull member_function_type_1)(int);
typedef int X::* _Nonnull member_data_type_1;
typedef nullptr_t _Nonnull nonnull_nullptr_t; // expected-error{{nullability specifier '_Nonnull' cannot be applied to non-pointer type 'nullptr_t' (aka 'std::nullptr_t')}}

// Nullability can move into member pointers (this is suppressing a warning).
typedef _Nonnull int (X::* member_function_type_2)(int);
typedef int (X::* _Nonnull member_function_type_3)(int);
typedef _Nonnull int X::* member_data_type_2;

// Adding non-null via a template.
template<typename T>
struct AddNonNull {
  typedef _Nonnull T type; // expected-error{{nullability specifier '_Nonnull' cannot be applied to non-pointer type 'int'}}
  // expected-error@-1{{nullability specifier '_Nonnull' cannot be applied to non-pointer type 'std::nullptr_t'}}
  // expected-error@-2{{nullability specifier '_Nonnull' cannot be applied to non-pointer type 'NotPtr'}}
};

typedef AddNonNull<int *>::type nonnull_int_ptr_1;
typedef AddNonNull<int * _Nullable>::type nonnull_int_ptr_2; // FIXME: check that it was overridden
typedef AddNonNull<nullptr_t>::type nonnull_int_ptr_3; // expected-note{{in instantiation of template class}}

typedef AddNonNull<int>::type nonnull_non_pointer_1; // expected-note{{in instantiation of template class 'AddNonNull<int>' requested here}}

// Nullability on C++ class types (smart pointers).
struct NotPtr{};
typedef AddNonNull<NotPtr>::type nonnull_non_pointer_2; // expected-note{{in instantiation}}
struct _Nullable SmartPtr{
  SmartPtr();
  SmartPtr(nullptr_t);
  SmartPtr(const SmartPtr&);
  SmartPtr(SmartPtr&&);
  SmartPtr &operator=(const SmartPtr&);
  SmartPtr &operator=(SmartPtr&&);
};
typedef AddNonNull<SmartPtr>::type nonnull_smart_pointer_1;
template<class> struct _Nullable SmartPtrTemplate{};
typedef AddNonNull<SmartPtrTemplate<int>>::type nonnull_smart_pointer_2;
namespace std { inline namespace __1 {
  template <class> class unique_ptr {};
  template <class> class function;
  template <class Ret, class... Args> class function<Ret(Args...)> {};
} }
typedef AddNonNull<std::unique_ptr<int>>::type nonnull_smart_pointer_3;
typedef AddNonNull<std::function<int()>>::type nonnull_smart_pointer_4;

class Derived : public SmartPtr {};
Derived _Nullable x; // expected-error {{'_Nullable' cannot be applied}}
class DerivedPrivate : private SmartPtr {};
DerivedPrivate _Nullable y; // expected-error {{'_Nullable' cannot be applied}}

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
void accepts_nonnull_6(SmartPtr _Nonnull);

void test_accepts_nonnull_null_pointer_literal(X *x) {
  accepts_nonnull_1(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
  accepts_nonnull_2(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
  (x->*accepts_nonnull_3)(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
  accepts_nonnull_4(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
  accepts_nonnull_5(0); // expected-warning{{null passed to a callee that requires a non-null argument}}

  accepts_nonnull_6(nullptr); // expected-warning{{null passed to a callee that requires a non-null argument}}
}

template<void FP(_Nonnull int*)> 
void test_accepts_nonnull_null_pointer_literal_template() {
  FP(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
}

template void test_accepts_nonnull_null_pointer_literal_template<&accepts_nonnull_4>(); // expected-note{{instantiation of function template specialization}}

void TakeNonnull(void *_Nonnull);
void TakeSmartNonnull(SmartPtr _Nonnull);
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
  nonnull = (void *_Nonnull)nullable; // explicit cast OK

  SmartPtr _Nullable s_nullable;
  SmartPtr _Nonnull s(s_nullable); // expected-warning{{implicit conversion from nullable pointer 'SmartPtr _Nullable' to non-nullable pointer type 'SmartPtr _Nonnull'}}
  SmartPtr _Nonnull s2{s_nullable}; // expected-warning{{implicit conversion from nullable pointer 'SmartPtr _Nullable' to non-nullable pointer type 'SmartPtr _Nonnull'}}
  SmartPtr _Nonnull s3 = {s_nullable}; // expected-warning{{implicit conversion from nullable pointer 'SmartPtr _Nullable' to non-nullable pointer type 'SmartPtr _Nonnull'}}
  SmartPtr _Nonnull s4 = s_nullable; // expected-warning{{implicit conversion from nullable pointer 'SmartPtr _Nullable' to non-nullable pointer type 'SmartPtr _Nonnull'}}
  SmartPtr _Nonnull s_nonnull;
  s_nonnull = s_nullable; // expected-warning{{implicit conversion from nullable pointer 'SmartPtr _Nullable' to non-nullable pointer type 'SmartPtr _Nonnull'}}
  s_nonnull = {s_nullable}; // no warning here - might be nice?
  TakeSmartNonnull(s_nullable); //expected-warning{{implicit conversion from nullable pointer 'SmartPtr _Nullable' to non-nullable pointer type 'SmartPtr _Nonnull}}
  TakeSmartNonnull(s_nonnull); // OK
  s_nonnull = (SmartPtr _Nonnull)s_nullable; // explicit cast OK
  s_nonnull = static_cast<SmartPtr _Nonnull>(s_nullable); // explicit cast OK
}

void *_Nullable ReturnNullable();
SmartPtr _Nullable ReturnSmartNullable();

void AssignAndInitNonNullFromFn() {
  void *_Nonnull p(ReturnNullable()); // expected-warning{{implicit conversion from nullable pointer 'void * _Nullable' to non-nullable pointer type 'void * _Nonnull'}}
  void *_Nonnull p2{ReturnNullable()}; // expected-warning{{implicit conversion from nullable pointer 'void * _Nullable' to non-nullable pointer type 'void * _Nonnull'}}
  void *_Nonnull p3 = {ReturnNullable()}; // expected-warning{{implicit conversion from nullable pointer 'void * _Nullable' to non-nullable pointer type 'void * _Nonnull'}}
  void *_Nonnull p4 = ReturnNullable(); // expected-warning{{implicit conversion from nullable pointer 'void * _Nullable' to non-nullable pointer type 'void * _Nonnull'}}
  void *_Nonnull nonnull;
  nonnull = ReturnNullable(); // expected-warning{{implicit conversion from nullable pointer 'void * _Nullable' to non-nullable pointer type 'void * _Nonnull'}}
  nonnull = {ReturnNullable()}; // expected-warning{{implicit conversion from nullable pointer 'void * _Nullable' to non-nullable pointer type 'void * _Nonnull'}}
  TakeNonnull(ReturnNullable()); //expected-warning{{implicit conversion from nullable pointer 'void * _Nullable' to non-nullable pointer type 'void * _Nonnull}}

  SmartPtr _Nonnull s(ReturnSmartNullable()); // expected-warning{{implicit conversion from nullable pointer 'SmartPtr _Nullable' to non-nullable pointer type 'SmartPtr _Nonnull'}}
  SmartPtr _Nonnull s2{ReturnSmartNullable()}; // expected-warning{{implicit conversion from nullable pointer 'SmartPtr _Nullable' to non-nullable pointer type 'SmartPtr _Nonnull'}}
  SmartPtr _Nonnull s3 = {ReturnSmartNullable()}; // expected-warning{{implicit conversion from nullable pointer 'SmartPtr _Nullable' to non-nullable pointer type 'SmartPtr _Nonnull'}}
  SmartPtr _Nonnull s4 = ReturnSmartNullable(); // expected-warning{{implicit conversion from nullable pointer 'SmartPtr _Nullable' to non-nullable pointer type 'SmartPtr _Nonnull'}}
  SmartPtr _Nonnull s_nonnull;
  s_nonnull = ReturnSmartNullable(); // expected-warning{{implicit conversion from nullable pointer 'SmartPtr _Nullable' to non-nullable pointer type 'SmartPtr _Nonnull'}}
  s_nonnull = {ReturnSmartNullable()};
  TakeSmartNonnull(ReturnSmartNullable()); // expected-warning{{implicit conversion from nullable pointer 'SmartPtr _Nullable' to non-nullable pointer type 'SmartPtr _Nonnull'}}
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
  auto nestedBad = [](int [2][_Nonnull 2]) {}; // expected-error {{nullability specifier '_Nonnull' cannot be applied to non-pointer type 'int[2]'}}

  auto withTypedef = [](INTS _Nonnull) {};
  withTypedef(nullptr); // expected-warning {{null passed to a callee that requires a non-null argument}}
  auto withTypedefBad = [](INTS _Nonnull[2]) {}; // expected-error {{nullability specifier '_Nonnull' cannot be applied to non-pointer type 'INTS' (aka 'int[4]')}}
}

void testNullabilityCompletenessWithTemplate() {
  Template<int*> tip;
}

namespace GH60344 {
class a;
template <typename b> using c = b _Nullable; // expected-error {{'_Nullable' cannot be applied to non-pointer type 'GH60344::a'}}
c<a>;  // expected-note {{in instantiation of template type alias 'c' requested here}}
}
