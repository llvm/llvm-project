void f() __attribute__((unavailable));
void g();
#define g() f()
#define h() f()
#define x x
#define h2() f()

#define h3()
#undef h3
