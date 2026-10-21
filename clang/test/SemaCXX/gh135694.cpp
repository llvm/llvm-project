// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -std=c++20 %s

namespace GH135694_1 {

void a(...);
template<typename T> void b() {
  decltype(a(void())) *p; // expected-error {{cannot pass expression of type 'void' to variadic function}}
}
void c() { b<void>(); }
}


namespace GH135694_2 {

void a(...);
template<typename T>
bool b() {
    return requires { a(a(1)); }; // expected-error {{cannot pass expression of type 'void' to variadic function}}
}
bool c() { return b<void>(); }

}

namespace GH135694_3 {

void a(...);
template<typename T, auto F> void b() {
  decltype(F(void())) *p; // expected-error {{cannot pass expression of type 'void' to variadic function}}
}
void c() { b<void, a>(); } // expected-note {{in instantiation}}
}
