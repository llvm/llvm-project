namespace N { template<typename T> struct A { friend int f(A); }; }
int a = f(N::A<int>());
