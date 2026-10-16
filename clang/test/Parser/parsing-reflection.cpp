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
    (void)(^^A);
    (void)(^^B); // expected-error {{unknown or unimplemented reflectable entity}}
    (void)(^^f); // expected-error {{unknown or unimplemented reflectable entity}}
}

auto a = ^^auto; // expected-error {{'auto' not allowed here}}
auto b = ^^decltype(auto); // expected-error {{'decltype(auto)' not allowed here}}
auto c = ^^int &*; // expected-error {{'type name' declared as a pointer to a reference}}
auto d = ^^int[-1]; // expected-error {{array size is negative}}
auto e = ^^void[3]; // expected-error {{array has incomplete element type 'void'}}

template <class... Ts> void unexpanded() {
  (void)^^Ts; // expected-error {{expression contains unexpanded parameter pack 'Ts'}}
}

template <class T> consteval auto missing() {
  return ^^typename T::missing; // expected-error {{no type named 'missing' in 'Empty'}}
}
struct Empty {};
auto missing_result = missing<Empty>(); // expected-note {{in instantiation of function template specialization 'missing<Empty>' requested here}}

template <int... Ns> void unexpanded_arrays() {
  (void)^^int[Ns]; // expected-error {{expression contains unexpanded parameter pack 'Ns'}}
}

namespace Names { struct Record {}; using Alias = int; }
namespace Imported {
using Names::Record; // expected-note {{using declaration}}
using Names::Alias; // expected-note {{using declaration}}
}
auto imported_record = ^^Imported::Record; // expected-error {{cannot reflect a name found through a using-declaration}}
auto imported_alias = ^^Imported::Alias; // expected-error {{cannot reflect a name found through a using-declaration}}

namespace PackIndexing {
template <unsigned I, class... Ts> consteval auto selected() {
  return ^^Ts...[I]; // expected-error {{invalid index 2 for pack 'Ts' of size 2}}
}
auto bad_index = selected<2, int, float>(); // expected-note {{in instantiation of function template specialization 'PackIndexing::selected<2U, int, float>' requested here}}
}

template <class T, class U> using Pair = int; // expected-note {{template parameter is declared here}}
template <class... Ts> consteval auto expanded_pair() {
  return ^^Pair<Ts...>; // expected-error {{pack expansion used as argument for non-pack parameter of alias template}}
}

template <class T> struct Class;
template <class... Ts> void unexpanded_class() {
  (void)^^Class<Ts>; // expected-error {{expression contains unexpanded parameter pack 'Ts'}}
}

template <class T> using Alias = int;
template <class... Ts> void unexpanded_alias() {
  (void)^^Alias<Ts>; // expected-error {{expression contains an unexpanded parameter pack}}
}
