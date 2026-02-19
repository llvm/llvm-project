// RUN: %clang_cc1 %s -std=c++26 -freflection -fsyntax-only -verify

struct A{};
namespace B{};
void f(){};

consteval void test()
{
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
    (void)(^^const void);
    (void)(^^decltype(nullptr));

    (void)(^^::); // expected-error {{unknown or unimplemented reflectable entity}}
    constexpr auto x = 1;
    (void)(^^x); // expected-error {{unknown or unimplemented reflectable entity}}
    (void)(^^A); // expected-error {{unknown or unimplemented reflectable entity}}
    (void)(^^B); // expected-error {{unknown or unimplemented reflectable entity}}
    (void)(^^f); // expected-error {{unknown or unimplemented reflectable entity}}
}
