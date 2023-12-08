struct A { ~A() throw(int); };
struct B { A a; };
