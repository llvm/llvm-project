// RUN: %clang_cc1 -std=c++11 -verify %s
// RUN: %clang_cc1 -std=c++2a -verify %s

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

// The substitution proceeds in lexical order and stops when a condition that
// causes deduction to fail is encountered.
#if __cplusplus > 201702L
namespace reversed_operator_substitution_order {
  struct X { X(int); };
  struct Y { Y(int); };
  struct Cat {};
  namespace no_adl {
    Cat operator<=>(Y, X);
    bool operator<(int, Cat);

    template<typename T> struct indirect_sizeof {
      static_assert(sizeof(T) != 0);
      static const auto value = sizeof(T);
    };

    // We should substitute into the construction of the X object before the
    // construction of the Y object, so this is a SFINAE case rather than a
    // hard error. This requires substitution to proceed in lexical order
    // despite the prior rewrite to
    //    0 < (Y(...) <=> X(...))
    template<typename T> float &f(
        decltype(
          X(sizeof(T)) < Y(indirect_sizeof<T>::value)
        )
    );
    template<typename T> int &f(...);
  }
  int &r = no_adl::f<void>(true);
  float &s = no_adl::f<int>(true);
}
#endif
