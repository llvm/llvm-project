// RUN: %clang_cc1 %s -fopenacc -verify

// Ensure that we are properly handling 'vardecl' when they are created during
// error recovery. The errors themselves aren't really relevant/necessary to the
// bug fix.
struct Thing{ };
struct pair {
  // expected-error@+2{{no member named 'T1'}}
  // expected-error@+1{{expected a qualified name after 'typename'}}
  template <typename enable_if<Thing::template T1<int>() &&
                                   !Thing::template T1<int>(),
    // expected-error@+4{{non-friend class member 'type' cannot have a qualified name}}
    // expected-error@+3{{type specifier is required}}
    // expected-error@+2{{non-static data member 'type' cannot be declared as a template}}
    // expected-error@+1{{no member named 'type' in the global namespace}}
                               bool>::type = false>
  // expected-error@+1{{expected '(' for function-style cast or type construction}}
  void func(void);
};
