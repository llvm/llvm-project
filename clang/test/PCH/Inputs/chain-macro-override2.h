#define f() g()
#undef g
#undef h
#define h() g()
int x;
#undef h2

int h3();
