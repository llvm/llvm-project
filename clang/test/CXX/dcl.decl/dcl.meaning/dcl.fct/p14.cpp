// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

template<typename T> struct identity;
template<typename ...Types> struct tuple;

template<typename T, typename U> struct is_same {
  static const bool value = false;
};

template<typename T> struct is_same<T, T> {
  static const bool value = true;
};

// There is a syntactic ambiguity when an ellipsis occurs at the end
// of a parameter-declaration-clause without a preceding comma. In
// this case, the ellipsis is parsed as part of the
// abstract-declarator if the type of the parameter names a template
// parameter pack that has not been expanded; otherwise, it is parsed
// as part of the parameter-declaration-clause.

template<typename T, typename ...Types>
struct X0 {
  typedef identity<T(Types...)> function_pack_1;
  typedef identity<T(Types......)> variadic_function_pack_1; // expected-warning {{varargs}} expected-note {{pack}} expected-note {{insert ','}}
  typedef identity<T(T...)> variadic_1;
  typedef tuple<T(Types, ...)...> template_arg_expansion_1;
};



// FIXME: Once function parameter packs are implemented, we can test all of the disambiguation
