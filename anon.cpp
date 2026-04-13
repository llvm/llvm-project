template<typename T>
void foo(T) {}

struct {} a;
struct {} f;
struct {} g;
struct {} x;
union {} y;
union {int mem; } z;
auto l{[]{}};
struct S {
    struct {} a;
    struct {} b;
};

void bar() {
    foo<decltype(f)>(f);
    foo<decltype(g)>(g);
    foo<decltype(x)>(x);
    foo<decltype(y)>(y);
    foo<decltype(z)>(z);
    foo<decltype(l)>(l);
    {
        struct {} g;
        foo<decltype(g)>(g);
        foo<decltype(a)>(a);
        foo<decltype(S::a)>(S{}.a);
    }
}

