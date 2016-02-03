// RUN: %clang_cc1 -std=c++11 -verify %s

struct Q { typedef int type; };

// "The substitution occurs in all types and expressions that are used in [...]
// template parameter declarations." In particular, we must substitute into the
// type of a parameter pack that is not a pack expansion, even if we know the
// corresponding argument pack is empty.
template<typename T, typename T::type...> void a(T);
int &a(...);
int &a_disabled = a(0);
int &a_enabled = a(Q()); // expected-error {{cannot bind to a temporary of type 'void'}}

template<typename T, template<typename T::type> class ...X> void b(T);
int &b(...);
int &b_disabled = b(0);
int &b_enabled = b(Q()); // expected-error {{cannot bind to a temporary of type 'void'}}

template<typename T, template<typename T::type...> class ...X> void c(T);
int &c(...);
int &c_disabled = c(0);
int &c_enabled = c(Q()); // expected-error {{cannot bind to a temporary of type 'void'}}
