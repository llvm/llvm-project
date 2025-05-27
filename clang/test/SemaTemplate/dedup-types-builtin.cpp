// RUN: %clang_cc1 %s -verify

template <typename...> struct TypeList;

// FIXME: better error message, note for the location of the builtin.
static_assert(__is_same(
  __builtin_dedup_types<TypeList, int, int*, int, double, float>,
  TypeList<int, int*, double, float>));

template <template<typename ...> typename Templ, typename ...Types>
struct Dependent {
  using empty_list = __builtin_dedup_types<Templ>;
  using same = __builtin_dedup_types<Templ, Types...>;
  using twice = __builtin_dedup_types<Templ, Types..., Types...>;
  using dep_only_types = __builtin_dedup_types<TypeList, Types...>;
  using dep_only_template = __builtin_dedup_types<Templ, int, double, int>;
}; 

// Check the reverse condition to make sure we see an error and not accidentally produced dependent expression.
static_assert(!__is_same(Dependent<TypeList>::empty_list, TypeList<>)); // expected-error {{static assertion failed}}
static_assert(!__is_same(Dependent<TypeList>::same, TypeList<>)); // expected-error {{static assertion failed}}
static_assert(!__is_same(Dependent<TypeList>::twice, TypeList<>)); // expected-error {{static assertion failed}}
static_assert(!__is_same(Dependent<TypeList>::dep_only_types, TypeList<>)); // expected-error {{static assertion failed}}
static_assert(!__is_same(Dependent<TypeList>::dep_only_template, TypeList<int, double>)); // expected-error {{static assertion failed}}
static_assert(!__is_same(Dependent<TypeList, int*, double*, int*>::empty_list, TypeList<>)); // expected-error {{static assertion failed}}
static_assert(!__is_same(Dependent<TypeList, int*, double*, int*>::same, TypeList<int*, double*>)); // expected-error {{static assertion failed}}
static_assert(!__is_same(Dependent<TypeList, int*, double*, int*>::twice, TypeList<int*, double*>)); // expected-error {{static assertion failed}}
static_assert(!__is_same(Dependent<TypeList, int*, double*, int*>::dep_only_types, TypeList<int*, double*>)); // expected-error {{static assertion failed}}
static_assert(!__is_same(Dependent<TypeList, int*, double*, int*>::dep_only_template, TypeList<int, double>)); // expected-error {{static assertion failed}}


template <class ...T>
using Twice = TypeList<T..., T...>;

// FIXME: move this test into a template, add a test that doing expansions outside of templates is an error.
static_assert(!__is_same(__builtin_dedup_types<Twice, int, double, int>, TypeList<int, double, int, double>)); // expected-error {{static assertion failed}}

template <int...> struct IntList;
// Wrong kinds of template arguments.
// FIXME: make the error message below point at this file.
__builtin_dedup_types<IntList, int>* wrong_template; // expected-error@* {{template argument for non-type template parameter must be an expression}}
                                                     // expected-note@* {{template template argument has different template parameters than its corresponding template template parameter}}
                                                     // expected-note@-5 {{template parameter is declared here}}
__builtin_dedup_types<TypeList, 1, 2, 3>* wrong_template_args; // expected-error  {{template argument for template type parameter must be a type}}
                                                               // expected-note@* {{template parameter from hidden source}}
__builtin_dedup_types<> not_enough_args; // expected-error {{too few template arguments}}
                                         // expected-note@* {{template declaration from hidden source}}
__builtin_dedup_types missing_template_args; // expected-error {{use of template '__builtin_dedup_types' requires template arguments}}
                                             // expected-note@* {{template declaration from hidden source}}

// Make sure various canonical / non-canonical type representations do not affect results
// of the deduplication and the qualifiers do end up creating different types when C++ requires it.
using Int = int;
using CInt = const Int;
using IntArray = Int[10];
using CIntArray = Int[10];
using IntPtr = int*;
using CIntPtr = const int*;

static_assert( 
  !__is_same( // expected-error {{static assertion failed}}
    __builtin_dedup_types<TypeList,
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
    >,
    TypeList<int, const int, int[10], const int [10], int*, int* const, const int*, int*const *, const int**, const int*const*>),
  "");
