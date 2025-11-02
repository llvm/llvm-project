// RUN: %clang_cc1 %s -std=c++26 -freflection -fsyntax-only -verify

namespace a {
struct X {
    int y;
    bool operator==(const X& other)
    {
        return y == other.y;
    }
};

namespace b {
    struct Y {};
    int x;
}

template<typename T>
struct Z{
    template<typename U>
    using type = U;
};

}

int main()
{
    (void)(^^::);
    (void)(^^void);
    (void)(^^bool);
    (void)(^^char);
    (void)(^^signed char);
    (void)(^^unsigned char);
    (void)(^^short);
    (void)(^^unsigned short);
    (void)(^^int);
    (void)(^^unsigned int);
    (void)(^^long);
    (void)(^^unsigned long);
    (void)(^^long long);
    (void)(^^float);
    (void)(^^double);

    // Not supported yet.
    (void)^^a; // expected-error {{expected reflectable entity}}
    (void)^^a::; // expected-error {{expected reflectable entity}}
    (void)^^a::b::X; // expected-error {{expected reflectable entity}}
    (void)^^a::X::; // expected-error {{expected reflectable entity}}
    (void)(^^a::b); // expected-error {{expected reflectable entity}}
    (void)^^a::b::; // expected-error {{expected reflectable entity}}
    (void)^^a::b::Y; // expected-error {{expected reflectable entity}}
    (void)^^a::b::x; // expected-error {{expected reflectable entity}}
    (void)^^a::b::Y::; // expected-error {{expected reflectable entity}}
    (void)(^^::a::); // expected-error {{expected reflectable entity}}
    (void)(^^::a::X::operator==); // expected-error {{expected reflectable entity}}
    (void)(^^::a::X::~X()); // expected-error {{expected reflectable entity}}
    (void)(^^::a::Z<int>); // expected-error {{expected reflectable entity}}
    (void)(^^::a::Z<int>::template type<int>); // expected-error {{expected reflectable entity}}
    namespace c = a::b;
    (void)(^^c); // expected-error {{expected reflectable entity}}
}
