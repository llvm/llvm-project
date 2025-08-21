// RUN: %clang_cc1 %s -verify
template <typename...> struct TypeList;

// === Check results of the builtin.
template <class>
struct TemplateWrapper {
  static_assert(__is_same( // expected-error {{static assertion contains an unexpanded parameter pack}}
    TypeList<__builtin_dedup_pack<int, int*, int, double, float>>,
    TypeList<int, int*, double, float>));
};

template <template<typename ...> typename Templ, typename ...Types>
struct Dependent {
  using empty_list = Templ<__builtin_dedup_pack<>...>;
  using same = Templ<__builtin_dedup_pack<Types...>...>;
  using twice = Templ<__builtin_dedup_pack<Types..., Types...>...>;
  using dep_only_types = TypeList<__builtin_dedup_pack<Types...>...>;
  using dep_only_template = Templ<__builtin_dedup_pack<int, double, int>...>;
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

template <class>
struct TwiceTemplateWrapper {
  static_assert(!__is_same(Twice<__builtin_dedup_pack<int, double, int>...>, TypeList<int, double, int, double>)); // expected-error {{static assertion failed}}

};
template struct TwiceTemplateWrapper<int>; // expected-note {{in instantiation of template class 'TwiceTemplateWrapper<int>' requested here}}

template <int...> struct IntList;
// Wrong kinds of template arguments.
template <class> struct IntListTemplateWrapper {
  IntList<__builtin_dedup_pack<int>...>* wrong_template; // expected-error {{template argument for non-type template parameter must be an expression}}
                                                         // expected-note@-4 {{template parameter is declared here}}
  TypeList<__builtin_dedup_pack<1, 2, 3>...>* wrong_template_args; // expected-error  {{template argument for template type parameter must be a type}}
                                                                    // expected-note@* {{template parameter from hidden source}}
  __builtin_dedup_pack<> not_enough_args; // expected-error {{data member type contains an unexpanded parameter pack}}
                                          // expected-note@* {{template declaration from hidden source}}
  __builtin_dedup_pack missing_template_args; // expected-error {{use of template '__builtin_dedup_pack' requires template arguments}}
};

// Make sure various canonical / non-canonical type representations do not affect results
// of the deduplication and the qualifiers do end up creating different types when C++ requires it.
using Int = int;
using CInt = const Int;
using IntArray = Int[10];
using CIntArray = Int[10];
using IntPtr = int*;
using CIntPtr = const int*;

template <class>
struct Foo {
  static_assert(
    !__is_same( // expected-error {{static assertion failed}}
                // expected-note@* {{in instantiation of template class 'Foo<int>'}}
      TypeList<__builtin_dedup_pack<
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
      >...>,
      TypeList<int, const int, int[10], const int [10], int*, int* const, const int*, int*const *, const int**, const int*const*>),
    "");
};

template struct Foo<int>;

// === Show an error when packs are used in non-template contexts.
static_assert(!__is_same(TypeList<__builtin_dedup_pack<int>...>, TypeList<int>)); // expected-error {{outside}}
// Non-dependent uses in template are fine, though.
template <class T>
struct NonDepInTemplate {
  static_assert(!__is_same(TypeList<__builtin_dedup_pack<int>...>, TypeList<int>)); // expected-error {{static assertion failed}}
};
template struct NonDepInTemplate<int>; // expected-note {{requested here}}

template <template<class...> class T = __builtin_dedup_pack> // expected-error {{use of template '__builtin_dedup_pack' requires template arguments}}
                                                             // expected-note@* {{template declaration from hidden source}}
struct UseAsTemplate;
template <template<class...> class>
struct AcceptsTemplateArg;
template <class>
struct UseAsTemplateWrapper {
  AcceptsTemplateArg<__builtin_dedup_pack>* a; // expected-error {{use of template '__builtin_dedup_pack' requires template arguments}}
                                               // expected-note@* {{template declaration from hidden source}}
};

// === Check how expansions in various contexts behave.
// The following cases are not supported yet, should produce an error.
template <class... T>
struct DedupBases : __builtin_dedup_pack<T...>... {};
struct Base1 {
   int a1;
};
struct Base2 {
   int a2;
};
static_assert(DedupBases<Base1, Base1, Base2, Base1, Base2, Base2>{1, 2}.a1 != 1); // expected-error {{static assertion failed}} \
                                                                                   // expected-note {{}}
static_assert(DedupBases<Base1, Base1, Base2, Base1, Base2, Base2>{1, 2}.a2 != 2); // expected-error {{static assertion failed}} \
                                                                                   // expected-note {{}}

template <class ...T>
constexpr int dedup_params(__builtin_dedup_pack<T...>... as) {
 return (as + ...);
}
static_assert(dedup_params<int, int, short, int, short, short>(1, 2)); // expected-error {{no matching function}} \
                                                                       // expected-note@-3 {{expansions of '__builtin_dedup_pack' are not supported here}}

template <class ...T>
constexpr int dedup_params_into_type_list(TypeList<__builtin_dedup_pack<T...>...> *, T... as) {
 return (as + ...);
}
static_assert(dedup_params_into_type_list(static_cast<TypeList<int,short,long>*>(nullptr), 1, short(1), 1, 1l, 1l) != 5); // expected-error {{static assertion failed}} \
                                                                                                                          // expected-note {{expression evaluates}}

template <class T, __builtin_dedup_pack<T, int>...> // expected-error 2{{expansions of '__builtin_dedup_pack' are not supported here}}
struct InTemplateParams {};
InTemplateParams<int> itp1;
InTemplateParams<int, 1, 2, 3, 4, 5> itp2;

template <class T>
struct DeepTemplateParams {
  template <__builtin_dedup_pack<T, int>...> // expected-error {{expansions of '__builtin_dedup_pack' are not supported here}}
  struct Templ {};
};
DeepTemplateParams<int>::Templ<> dtp1; // expected-note {{requested here}} \
                                       // expected-error {{no template named 'Templ'}}


template <class ...T>
struct MemInitializers : T... {
  MemInitializers() : __builtin_dedup_pack<T...>()... {} // expected-error 2{{expansions of '__builtin_dedup_pack' are not supported here.}}
};
MemInitializers<> mi1; // expected-note {{in instantiation of member function}}
MemInitializers<Base1, Base2> mi2; // expected-note {{in instantiation of member function}}

template <class ...T>
constexpr int dedup_in_expressions() {
 // counts the number of unique Ts.
 return ((1 + __builtin_dedup_pack<T...>()) + ...); // expected-error {{expansions of '__builtin_dedup_pack' are not supported here.}} \
                                                    // expected-note@+3 {{in instantiation of function template specialization}}
}
static_assert(dedup_in_expressions<int, int, short, double, int, short, double, int>() == 3); // expected-error {{not an integral constant expression}}

template <class ...T>
void in_exception_spec() throw(__builtin_dedup_pack<T...>...); // expected-error{{C++17 does not allow dynamic exception specifications}} \
                                                               // expected-note {{use 'noexcept}} \
                                                               // expected-error{{expansions of '__builtin_dedup_pack' are not supported here.}}

void test_in_exception_spec() {
  in_exception_spec<int, double, int>(); // expected-note {{instantiation of exception specification}}
}

template <class ...T>
constexpr bool in_type_trait = __is_trivially_constructible(int, __builtin_dedup_pack<T...>...);  // expected-error{{expansions of '__builtin_dedup_pack' are not supported here.}}

static_assert(in_type_trait<int, int, int>); // expected-note{{in instantiation of variable template specialization}}

template <class ...T>
struct InFriends {
  friend __builtin_dedup_pack<T>...; // expected-warning {{variadic 'friend' declarations are a C++2c extension}} \
                                     // expected-error  2 {{expansions of '__builtin_dedup_pack' are not supported here.}} \
                                     // expected-note@* 2 {{in instantiation of template class}}

};
struct Friend1 {};
struct Friend2 {};
InFriends<> if1;
InFriends<Friend1, Friend2> if2;

template <class ...T>
struct InUsingDecl {
  using __builtin_dedup_pack<T...>::func...; // expected-error  2 {{expansions of '__builtin_dedup_pack' are not supported here.}}
};
struct WithFunc1 { void func(); };
struct WithFunc2 { void func(int); };
InUsingDecl<> iu1; // expected-note {{in instantiation of template class}}
InUsingDecl<WithFunc1, WithFunc2> iu2; // expected-note {{in instantiation of template class}}

// Note: produces parsing errors and does not construct pack indexing.
// Keep this commented out until the parser supports this.
//
// template <class ...T>
// struct InPackIndexing {
//
//   using type = __builtin_dedup_pack<T...>...[0];
// };
// static_assert(__is_same(InPackIndexing<int, int>, int));

template <class ...T>
struct LambdaInitCaptures {
  static constexpr int test() {
    [...foos=__builtin_dedup_pack<T...>()]{}; // expected-warning {{initialized lambda pack captures are a C++20 extension}} \
                                              // expected-error 2{{expansions of '__builtin_dedup_pack' are not supported here.}}
    return 3;
  }
};
static_assert(LambdaInitCaptures<>::test() == 3); // expected-note {{in instantiation of member function}}
static_assert(LambdaInitCaptures<int, int, int>::test() == 3); // expected-note {{in instantiation of member function}}

template <class ...T>
struct alignas(__builtin_dedup_pack<T...>...) AlignAs {}; // expected-error 2{{expansions of '__builtin_dedup_pack' are not supported here.}}
AlignAs<> aa1; // expected-note {{in instantiation of template class}}
AlignAs<int, double> aa2; // expected-note {{in instantiation of template class}}

