struct A { ~A() throw(int); };
struct B { A a; };
inline void f(B *p) { p->~B(); }
