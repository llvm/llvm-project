namespace N { template<typename T> struct A { friend int f(A); }; }
// It would seem like this variable should be called 'c'.
// But that makes the original problem disappear...
int e = f(N::A<int>());
#include "a.h"
#include "b.h"
