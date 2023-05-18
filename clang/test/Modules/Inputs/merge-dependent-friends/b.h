namespace N { template<typename T> struct A { friend int f(A); }; }
int b = f(N::A<int>());
