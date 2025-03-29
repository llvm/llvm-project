// RUN: %clang_cc1 %s -verify

template <typename...> struct TypeList;

static_assert(__is_same(
  TypeList<__builtin_dedup_types<int, int*, int, double, float>>,
  TypeList<int, int*, double, float>));

template <template<typename ...> typename Templ, typename ...Types>
struct Dependent {
  using empty_list = Templ<__builtin_dedup_types<>>;
  using same = Templ<__builtin_dedup_types<Types...>>;
  using twice = Templ<__builtin_dedup_types<Types..., Types...>>;
  using dep_only_types = TypeList<__builtin_dedup_types<Types...>>;
  using dep_only_template = Templ<__builtin_dedup_types<int, double, int>>;
}; 

static_assert(__is_same(Dependent<TypeList>::empty_list, TypeList<>));
static_assert(__is_same(Dependent<TypeList>::same, TypeList<>));
static_assert(__is_same(Dependent<TypeList>::twice, TypeList<>));
static_assert(__is_same(Dependent<TypeList>::dep_only_types, TypeList<>));
static_assert(__is_same(Dependent<TypeList>::dep_only_template, TypeList<int, double>));

static_assert(__is_same(Dependent<TypeList, int*, double*, int*>::empty_list, TypeList<>));
static_assert(__is_same(Dependent<TypeList, int*, double*, int*>::same, TypeList<int*, double*>));
static_assert(__is_same(Dependent<TypeList, int*, double*, int*>::twice, TypeList<int*, double*>));
static_assert(__is_same(Dependent<TypeList, int*, double*, int*>::dep_only_types, TypeList<int*, double*>));
static_assert(__is_same(Dependent<TypeList, int*, double*, int*>::dep_only_template, TypeList<int, double>));


template <class ...T>
using Twice = TypeList<T..., T...>;

static_assert(__is_same(Twice<__builtin_dedup_types<int, double, int>>, TypeList<int, double, int, double>));


template <int...> struct IntList;
// Wrong kinds of template arguments.
// FIXME: make the error message below point at this file.
IntList<__builtin_dedup_types<int>>* wrong_template; // expected-error@* {{template argument for non-type template parameter must be an expression}}
                                                         // expected-note@-4 {{template parameter is declared here}}
TypeList<__builtin_dedup_types<1, 2, 3>>* wrong_template_args; // expected-error  {{template argument for template type parameter must be a type}}
                                                           // expected-note@* {{template parameter from hidden source}}
__builtin_dedup_types<> not_enough_args; // expected-error {{'__builtin_dedup_types' can only be used directly inside a template argument list}}
                                             // expected-note@* {{template declaration from hidden source}}
__builtin_dedup_types missing_template_args; // expected-error {{use of template '__builtin_dedup_types' requires template arguments}}

// Make sure various canonical / non-canonical type representations do not affect results
// of the deduplication and the qualifiers do end up creating different types when C++ requires it.
using Int = int;
using CInt = const Int;
using IntArray = Int[10];
using CIntArray = Int[10];
using IntPtr = int*;
using CIntPtr = const int*;
static_assert(
  __is_same(
    TypeList<__builtin_dedup_types<
      Int, int,
      const int, const Int, CInt, const CInt,
      IntArray, Int[10], int[10],
      const IntArray, const int[10], CIntArray, const CIntArray,
      IntPtr, int*,
      const IntPtr, int* const,
      CIntPtr, const int*,
      const IntPtr*, int*const*,
      CIntPtr*, const int**,
      const CIntPtr*, const int* const*
    >>,
    TypeList<int, const int, int[10], const int [10], int*, int* const, const int*, int*const *, const int**, const int*const*>),
  "");

// FIXME: tests for locations of template arguments, ideally we should point into the original locations of the template arguments.
