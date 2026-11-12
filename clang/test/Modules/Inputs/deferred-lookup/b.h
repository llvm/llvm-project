namespace N { template<typename T> struct A { friend int f(A); }; }
namespace N { int f(int); }
namespace N { int f(int); }
#include "a.h"
namespace N { int f(int); }
inline int g() { return f(N::A<int>()); }
