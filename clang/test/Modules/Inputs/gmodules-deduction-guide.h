struct A {
};

template <class T>
struct S{
  S(const A &);
};

S(const A&) -> S<A>;

typedef decltype(S(A())) Type0;
