// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

// Example bind implementation from the variadic templates proposal,
// ISO C++ committee document number N2080.

// Helper type traits
template<typename T>
struct add_reference {
  typedef T &type;
};

template<typename T>
struct add_reference<T&> {
  typedef T &type;
};

template<typename T>
struct add_const_reference {
  typedef T const &type;
};

template<typename T>
struct add_const_reference<T&> {
  typedef T &type;
};

template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

template<typename T> 
class reference_wrapper { 
  T *ptr;

public:
  reference_wrapper(T& t) : ptr(&t) { }
  operator T&() const { return *ptr; }
};

template<typename T> reference_wrapper<T> ref(T& t) { 
  return reference_wrapper<T>(t); 
}
template<typename T> reference_wrapper<const T> cref(const T& t) {
  return reference_wrapper<const T>(t); 
}

template<typename... Values> class tuple;

// Basis case: zero-length tuple
template<> class tuple<> { };

template<typename Head, typename... Tail> 
class tuple<Head, Tail...> : private tuple<Tail...> { 
  typedef tuple<Tail...> inherited;

public: 
  tuple() { }
  // implicit copy-constructor is okay

  // Construct tuple from separate arguments. 
  tuple(typename add_const_reference<Head>::type v,
        typename add_const_reference<Tail>::type... vtail) 
    : m_head(v), inherited(vtail...) { }

  // Construct tuple from another tuple. 
  template<typename... VValues> tuple(const tuple<VValues...>& other)
    : m_head(other.head()), inherited(other.tail()) { }

  template<typename... VValues> tuple& 
  operator=(const tuple<VValues...>& other) {
    m_head = other.head(); 
    tail() = other.tail(); 
    return *this;
  }

  typename add_reference<Head>::type head() { return m_head; } 
  typename add_reference<const Head>::type head() const { return m_head; }
  inherited& tail() { return *this; } 
  const inherited& tail() const { return *this; }

protected: 
  Head m_head;
};

// Creation functions
template<typename T> 
struct make_tuple_result {
  typedef T type;
};

template<typename T> 
struct make_tuple_result<reference_wrapper<T> > {
  typedef T& type;
};

template<typename... Values> 
tuple<typename make_tuple_result<Values>::type...> 
make_tuple(const Values&... values) {
  return tuple<typename make_tuple_result<Values>::type...>(values...);
}

template<typename... Values> 
tuple<Values&...> tie(Values&... values) {
  return tuple<Values&...>(values...);
}

// Helper classes
template<typename Tuple> struct tuple_size;

template<typename... Values> struct tuple_size<tuple<Values...> > {
  static const int value = sizeof...(Values);
};

template<int I, typename Tuple> struct tuple_element;

template<int I, typename Head, typename... Tail> 
struct tuple_element<I, tuple<Head, Tail...> > {
  typedef typename tuple_element<I-1, tuple<Tail...> >::type type;
};

template<typename Head, typename... Tail> 
struct tuple_element<0, tuple<Head, Tail...> > {
  typedef Head type;
};

// Element access
template<int I, typename Tuple> class get_impl;
template<int I, typename Head, typename... Values> 
class get_impl<I, tuple<Head, Values...> > {
  typedef typename tuple_element<I-1, tuple<Values...> >::type Element;
  typedef typename add_reference<Element>::type RJ; 
  typedef typename add_const_reference<Element>::type PJ;
  typedef get_impl<I-1, tuple<Values...> > Next;
public: 
  static RJ get(tuple<Head, Values...>& t) { return Next::get(t.tail()); }
  static PJ get(const tuple<Head, Values...>& t) { return Next::get(t.tail()); }
};

template<typename Head, typename... Values> 
class get_impl<0, tuple<Head, Values...> > {
  typedef typename add_reference<Head>::type RJ; 
  typedef typename add_const_reference<Head>::type PJ;
public: 
  static RJ get(tuple<Head, Values...>& t) { return t.head(); } 
  static PJ get(const tuple<Head, Values...>& t) { return t.head(); }
};

template<int I, typename... Values> typename add_reference<
typename tuple_element<I, tuple<Values...> >::type >::type
get(tuple<Values...>& t) { 
  return get_impl<I, tuple<Values...> >::get(t);
}

template<int I, typename... Values> typename add_const_reference<
typename tuple_element<I, tuple<Values...> >::type >::type
get(const tuple<Values...>& t) { 
  return get_impl<I, tuple<Values...> >::get(t);
}

// Relational operators
inline bool operator==(const tuple<>&, const tuple<>&) { return true; }

template<typename T, typename... TTail, typename U, typename... UTail> 
bool operator==(const tuple<T, TTail...>& t, const tuple<U, UTail...>& u) {
  return t.head() == u.head() && t.tail() == u.tail();
}

template<typename... TValues, typename... UValues> 
bool operator!=(const tuple<TValues...>& t, const tuple<UValues...>& u) {
  return !(t == u); 
}

inline bool operator<(const tuple<>&, const tuple<>&) { return false; }

template<typename T, typename... TTail, typename U, typename... UTail> 
bool operator<(const tuple<T, TTail...>& t, const tuple<U, UTail...>& u) {
  return (t.head() < u.head() || (!(t.head() < u.head()) && t.tail() < u.tail()));
}

template<typename... TValues, typename... UValues> 
bool operator>(const tuple<TValues...>& t, const tuple<UValues...>& u) {
  return u < t;
}

template<typename... TValues, typename... UValues>
bool operator<=(const tuple<TValues...>& t, const tuple<UValues...>& u) {
  return !(u < t);
}

template<typename... TValues, typename... UValues>
bool operator>=(const tuple<TValues...>& t, const tuple<UValues...>& u) {
  return !(t < u);
}

// make_indices helper
template<int...> struct int_tuple {};
// make_indexes impl is a helper for make_indexes 
template<int I, typename IntTuple, typename... Types> struct make_indexes_impl;

template<int I, int... Indexes, typename T, typename... Types>
struct make_indexes_impl<I, int_tuple<Indexes...>, T, Types...> {
  typedef typename make_indexes_impl<I+1, int_tuple<Indexes..., I>, Types...>::type type;
};

template<int I, int... Indexes> 
struct make_indexes_impl<I, int_tuple<Indexes...> > {
  typedef int_tuple<Indexes...> type; 
};

template<typename... Types>
struct make_indexes : make_indexes_impl<0, int_tuple<>, Types...> { 
}; 

// Bind
template<typename T> struct is_bind_expression {
  static const bool value = false;
};

template<typename T> struct is_placeholder {
  static const int value = 0;
};


template<typename F, typename... BoundArgs> class bound_functor {
  typedef typename make_indexes<BoundArgs...>::type indexes; 
public:
  typedef typename F::result_type result_type; 
  explicit bound_functor(const F& f, const BoundArgs&... bound_args)
    : f(f), bound_args(bound_args...) { } template<typename... Args>
  typename F::result_type operator()(Args&... args);
private: F f;
  tuple<BoundArgs...> bound_args;
};

template<typename F, typename... BoundArgs> 
inline bound_functor<F, BoundArgs...> bind(const F& f, const BoundArgs&... bound_args) {
  return bound_functor<F, BoundArgs...>(f, bound_args...);
}

template<typename F, typename ...BoundArgs>
struct is_bind_expression<bound_functor<F, BoundArgs...> > {
  static const bool value = true;
};

// enable_if helper
template<bool Cond, typename T = void>
struct enable_if;

template<typename T>
struct enable_if<true, T> {
  typedef T type;
};

template<typename T>
struct enable_if<false, T> { };

// safe_tuple_element helper
template<int I, typename Tuple, typename = void>
struct safe_tuple_element { };

template<int I, typename... Values> 
struct safe_tuple_element<I, tuple<Values...>,
                          typename enable_if<(I >= 0 && I < tuple_size<tuple<Values...> >::value)>::type> { 
   typedef typename tuple_element<I, tuple<Values...> >::type type;
};

// mu
template<typename Bound, typename... Args> 
inline typename safe_tuple_element<is_placeholder<Bound>::value -1,
                                   tuple<Args...> >::type 
mu(Bound& bound_arg, const tuple<Args&...>& args) {
  return get<is_placeholder<Bound>::value-1>(args);
}

template<typename T, typename... Args> 
inline T& mu(reference_wrapper<T>& bound_arg, const tuple<Args&...>&) {
  return bound_arg.get();
}

template<typename F, int... Indexes, typename... Args> 
inline typename F::result_type 
unwrap_and_forward(F& f, int_tuple<Indexes...>, const tuple<Args&...>& args) {
  return f(get<Indexes>(args)...);
}

template<typename Bound, typename... Args> 
inline typename enable_if<is_bind_expression<Bound>::value,
                          typename Bound::result_type>::type 
mu(Bound& bound_arg, const tuple<Args&...>& args) {
  typedef typename make_indexes<Args...>::type Indexes; 
  return unwrap_and_forward(bound_arg, Indexes(), args);
}

template<typename T>
struct is_reference_wrapper {
  static const bool value = false;
};

template<typename T>
struct is_reference_wrapper<reference_wrapper<T>> {
  static const bool value = true;
};

template<typename Bound, typename... Args> 
inline typename enable_if<(!is_bind_expression<Bound>::value 
                           && !is_placeholder<Bound>::value 
                           && !is_reference_wrapper<Bound>::value), 
                           Bound&>::type
mu(Bound& bound_arg, const tuple<Args&...>&) {
  return bound_arg;
}

template<typename F, typename... BoundArgs, int... Indexes, typename... Args> 
typename F::result_type apply_functor(F& f, tuple<BoundArgs...>& bound_args, 
                                      int_tuple<Indexes...>, 
                                      const tuple<Args&...>& args) {
  return f(mu(get<Indexes>(bound_args), args)...);
}

template<typename F, typename... BoundArgs> 
template<typename... Args> 
typename F::result_type bound_functor<F, BoundArgs...>::operator()(Args&... args) {
  return apply_functor(f, bound_args, indexes(), tie(args...));
}

template<int N> struct placeholder { };
template<int N>
struct is_placeholder<placeholder<N>> {
  static const int value = N;
};

template<typename T>
struct plus {
  typedef T result_type;

  T operator()(T x, T y) { return x + y; }
};

placeholder<1> _1;

// Test bind
void test_bind() {
  int x = 17;
  int y = 25;
  bind(plus<int>(), x, _1)(y);
}
