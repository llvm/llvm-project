// RUN: %clang_cc1 %s -std=c++26 -freflection -fsyntax-only -verify

namespace a {
struct T {};
namespace b {
struct U {};
int x;
}
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
    (void)^^a::b::T; // expected-error {{expected reflectable entity}}
    (void)^^a::T::; // expected-error {{expected reflectable entity}}
    (void)(^^a::b); // expected-error {{expected reflectable entity}}
    (void)^^a::b::; // expected-error {{expected reflectable entity}}
    (void)^^a::b::U; // expected-error {{expected reflectable entity}}
    (void)^^a::b::x; // expected-error {{expected reflectable entity}}
    (void)^^a::b::U::; // expected-error {{expected reflectable entity}}
}
