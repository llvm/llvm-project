// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// Metafunction to extract the Nth type from a set of types.
template<unsigned N, typename ...Types> struct get_nth_type;

template<unsigned N, typename Head, typename ...Tail>
struct get_nth_type<N, Head, Tail...> : get_nth_type<N-1, Tail...> { };

template<typename Head, typename ...Tail>
struct get_nth_type<0, Head, Tail...> {
  typedef Head type;
};

// Placeholder type  when get_nth_type fails.
struct no_type {};

template<unsigned N>
struct get_nth_type<N> {
  typedef no_type type;
};

template<typename T, typename U> struct pair { };
template<typename T, typename U> pair<T, U> make_pair(T, U);

// For a function parameter pack that occurs at the end of the
// parameter-declaration-list, the type A of each remaining argument
// of the call is compared with the type P of the declarator-id of the
// function parameter pack.
template<typename ...Args>
typename get_nth_type<0, Args...>::type first_arg(Args...);

template<typename ...Args>
typename get_nth_type<1, Args...>::type second_arg(Args...);

void test_simple_deduction(int *ip, float *fp, double *dp) {
  int *ip1 = first_arg(ip);
  int *ip2 = first_arg(ip, fp);
  int *ip3 = first_arg(ip, fp, dp);
  no_type nt1 = first_arg();
}

template<typename ...Args>
typename get_nth_type<0, Args...>::type first_arg_ref(Args&...);

template<typename ...Args>
typename get_nth_type<1, Args...>::type second_arg_ref(Args&...);

void test_simple_ref_deduction(int *ip, float *fp, double *dp) {
  int *ip1 = first_arg_ref(ip);
  int *ip2 = first_arg_ref(ip, fp);
  int *ip3 = first_arg_ref(ip, fp, dp);
  no_type nt1 = first_arg_ref();
}


template<typename ...Args1, typename ...Args2>
typename get_nth_type<0, Args1...>::type first_arg_pair(pair<Args1, Args2>...); // expected-note{{candidate template ignored: could not match 'pair<Args1, Args2>' against 'int'}}

template<typename ...Args1, typename ...Args2>
typename get_nth_type<1, Args1...>::type second_arg_pair(pair<Args1, Args2>...);

void test_pair_deduction(int *ip, float *fp, double *dp) {
  int *ip1 = first_arg_pair(make_pair(ip, 17));
  int *ip2 = first_arg_pair(make_pair(ip, 17), make_pair(fp, 17));
  int *ip3 = first_arg_pair(make_pair(ip, 17), make_pair(fp, 17), 
                            make_pair(dp, 17));
  float *fp1 = second_arg_pair(make_pair(ip, 17), make_pair(fp, 17));
  float *fp2 = second_arg_pair(make_pair(ip, 17), make_pair(fp, 17), 
                               make_pair(dp, 17));
  no_type nt1 = first_arg_pair();
  no_type nt2 = second_arg_pair();
  no_type nt3 = second_arg_pair(make_pair(ip, 17));


  first_arg_pair(make_pair(ip, 17), 16); // expected-error{{no matching function for call to 'first_arg_pair'}}
}

// A function parameter pack not at the end of the parameter list is never
// deduced. We interpret this as meaning the types within it are never
// deduced, and thus must match explicitly-specified values.
template<typename ...Types> struct tuple { };

template<typename ...Types>
void pack_not_at_end(tuple<Types...>, Types... values, int); // expected-note {{<int *, double *> vs. <int, int>}}

void test_pack_not_at_end(tuple<int*, double*> t2) {
  pack_not_at_end(t2, 0, 0, 0); // expected-error {{no match}}
  // FIXME: Should the "original argument type must match deduced parameter
  // type" rule apply here?
  pack_not_at_end<int*, double*>(t2, 0, 0, 0); // ok
}
