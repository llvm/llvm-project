template<int N> struct A;
template<> struct A<1>;

template<int N> constexpr void f();
template<> constexpr void f<1>();

template<int N> extern int v;
template<> extern int v<1>;
