// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

// Example tuple implementation from the variadic templates proposal,
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

void test_tuple() {
  tuple<> t0a;
  tuple<> t0b(t0a);
  t0a = t0b;

  tuple<int> t1a;
  tuple<int> t1b(17);
  tuple<int> t1c(t1b);
  t1a = t1b;

  tuple<float> t1d(3.14159);
  tuple<float> t1e(t1d);
  t1d = t1e;

  int i;
  float f;
  double d;
  tuple<int*, float*, double*> t3a(&i, &f, &d);
}

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

template<typename T> const T *addr(const T& ref) { return &ref; }
void test_creation_functions() {
  int i;
  float f;
  double d;
  const tuple<int, float&, const double&> *t3p = addr(make_tuple(i, ref(f), cref(d)));
  const tuple<int&, float&, double&> *t3q = addr(tie(i, f, d));
}

// Helper classes
template<typename Tuple> struct tuple_size;

template<typename... Values> struct tuple_size<tuple<Values...> > {
  static const int value = sizeof...(Values);
};

int check_tuple_size_0[tuple_size<tuple<> >::value == 0? 1 : -1];
int check_tuple_size_1[tuple_size<tuple<int>>::value == 1? 1 : -1];
int check_tuple_size_2[tuple_size<tuple<float, double>>::value == 2? 1 : -1];
int check_tuple_size_3[tuple_size<tuple<char, unsigned char, signed char>>::value == 3? 1 : -1];

template<int I, typename Tuple> struct tuple_element;

template<int I, typename Head, typename... Tail> 
struct tuple_element<I, tuple<Head, Tail...> > {
  typedef typename tuple_element<I-1, tuple<Tail...> >::type type;
};

template<typename Head, typename... Tail> 
struct tuple_element<0, tuple<Head, Tail...> > {
  typedef Head type;
};

int check_tuple_element_0[is_same<tuple_element<0, tuple<int&, float, double>>::type,
                                  int&>::value? 1 : -1];

int check_tuple_element_1[is_same<tuple_element<1, tuple<int&, float, double>>::type,
                                  float>::value? 1 : -1];

int check_tuple_element_2[is_same<tuple_element<2, tuple<int&, float, double>>::type,
                                  double>::value? 1 : -1];

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

void test_element_access(tuple<int*, float*, double*&> t3) {
  int i;
  float f;
  double d;
  get<0>(t3) = &i;
  get<1>(t3) = &f;
  get<2>(t3) = &d;
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

void test_relational_operators(tuple<int*, float*, double*> t3) {
  (void)(t3 == t3);
  (void)(t3 != t3);
  (void)(t3 < t3);
  (void)(t3 <= t3);
  (void)(t3 >= t3);
  (void)(t3 > t3);
};
