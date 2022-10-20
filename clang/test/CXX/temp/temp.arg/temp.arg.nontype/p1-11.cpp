// RUN: %clang_cc1 -std=c++11 %s -verify -triple x86_64-linux-gnu

namespace std {
  typedef decltype(nullptr) nullptr_t;
}

template<int *ip> struct IP {  // expected-note 6 {{template parameter is declared here}}
  IP<ip> *ip2;
};

template<int &ip> struct IR {};

constexpr std::nullptr_t get_nullptr() { return nullptr; }

constexpr std::nullptr_t np = nullptr;

std::nullptr_t nonconst_np; // expected-note{{declared here}}

thread_local int tl; // expected-note {{refers here}}

IP<0> ip0; // expected-error{{null non-type template argument must be cast to template parameter type 'int *'}}
IP<(0)> ip1; // expected-error{{null non-type template argument must be cast to template parameter type 'int *'}}
IP<nullptr> ip2;
IP<get_nullptr()> ip3;
IP<(int*)0> ip4;
IP<np> ip5;
IP<nonconst_np> ip5; // expected-error{{non-type template argument of type 'std::nullptr_t' is not a constant expression}} \
// expected-note{{read of non-constexpr variable 'nonconst_np' is not allowed in a constant expression}}
IP<(float*)0> ip6; // expected-error{{null non-type template argument of type 'float *' does not match template parameter of type 'int *'}}
IP<&tl> ip7; // expected-error{{non-type template argument of type 'int *' is not a constant expression}}
IP<(int*)1> ip8; // expected-error {{non-type template argument '(int *)1' is invalid}}

IR<tl> ir1; // expected-error{{non-type template argument refers to thread-local object}}

struct X { };
template<int X::*pm> struct PM { // expected-note 2 {{template parameter is declared here}}
  PM<pm> *pm2;
};

PM<0> pm0; // expected-error{{null non-type template argument must be cast to template parameter type 'int X::*'}}
PM<(0)> pm1; // expected-error{{null non-type template argument must be cast to template parameter type 'int X::*'}}
PM<nullptr> pm2;
PM<get_nullptr()> pm3;
PM<(int X::*)0> pm4;
PM<np> pm5;

template<int (X::*pmf)(int)> struct PMF { // expected-note 2 {{template parameter is declared here}}
  PMF<pmf> *pmf2;
};

PMF<0> pmf0; // expected-error{{null non-type template argument must be cast to template parameter type 'int (X::*)(int)'}}
PMF<(0)> pmf1; // expected-error{{null non-type template argument must be cast to template parameter type 'int (X::*)(int)'}}
PMF<nullptr> pmf2;
PMF<get_nullptr()> pmf3;
PMF<(int (X::*)(int))0> pmf4;
PMF<np> pmf5;


template<std::nullptr_t np> struct NP { // expected-note 2{{template parameter is declared here}}
  NP<np> *np2;
};

NP<nullptr> np1;
NP<np> np2;
NP<get_nullptr()> np3;
NP<0> np4; // expected-error{{null non-type template argument must be cast to template parameter type 'std::nullptr_t'}}
constexpr int i = 7;
NP<i> np5; // expected-error{{non-type template argument of type 'const int' cannot be converted to a value of type 'std::nullptr_t'}}
