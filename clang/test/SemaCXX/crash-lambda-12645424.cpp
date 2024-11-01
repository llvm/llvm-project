// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify

// Crash due to a double-free

template<typename _Tp> struct __add_lvalue_reference_helper {};
template<typename _Tp> struct add_lvalue_reference :  __add_lvalue_reference_helper<_Tp> {
  typedef _Tp type;
};

template<typename... Types> struct type_list;
template<typename , template<typename> class... Funs> struct C;

template<typename T> struct C<T> {
	typedef T type;
};

template<typename T, template<typename>  class Fun0, template<typename> class... Funs> struct C<T, Fun0, Funs...> {
  typedef  typename C<typename Fun0<T>::type, Funs...>::type type;
};

template<class , template<typename> class... Funs> struct tl_map;
template<typename... Ts, template<typename> class... Funs> struct tl_map<type_list<Ts...>, Funs...> {
  typedef type_list<typename C<Ts, Funs...>::type...> type;
};

template<   class Pattern> struct F {
 typedef Pattern  filtered_pattern;
  tl_map< filtered_pattern, add_lvalue_reference > type;
};

template<class, class Pattern> struct get_case {
  F<Pattern> type;
};

template<class Pattern> struct rvalue_builder {
  template<typename Expr> typename get_case<Expr, Pattern>::type operator>>(Expr ); // expected-note {{candidate template ignored}}
};
  
template<typename Arg0> rvalue_builder< type_list<Arg0> > on(const Arg0& ) ;

class Z {
  int empty = on(0) >> [] {}; // expected-error {{invalid operands to binary expression}}
};
