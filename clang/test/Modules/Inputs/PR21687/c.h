#include "a.h"
inline void f() { X x, y(x); }
#include "b.h"
X x, y(x);
