// Header for PCH test cxx-offsetof-base.cpp

struct Base { int x; };
struct Derived : Base { int y; };
int o = __builtin_offsetof(Derived, x);
