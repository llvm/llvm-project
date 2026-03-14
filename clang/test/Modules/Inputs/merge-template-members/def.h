template<typename> struct A { int n; };
template<typename> struct B { typedef A<void> C; };
