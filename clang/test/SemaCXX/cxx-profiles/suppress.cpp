// RUN: %clang_cc1 -std=c++20 -fexperimental-cxx-profiles -fsyntax-only -verify %s

[[profiles::enforce(std::casting)]];

void statements(int *ip, const int *cip) {
  [[profiles::suppress(std::casting)]] (void)reinterpret_cast<char *>(ip);

  [[profiles::suppress(std::casting, rule: reinterpret_cast_char)]]
  (void)reinterpret_cast<char *>(ip);

  [[profiles::suppress(std::casting, rule: "reinterpret_cast_char",
                       justification: "byte access")]]
  (void)reinterpret_cast<char *>(ip);

  // Suppressing one rule does not suppress the others.
  [[profiles::suppress(std::casting, rule: const_cast)]]
  (void)reinterpret_cast<char *>(ip);
  // expected-error@-1 {{std::casting forbids reinterpret_cast from 'int *' to 'char *'}}

  [[profiles::suppress(std::casting, rule: reinterpret_cast_char)]]
  (void)(char *)cip;
  // expected-error@-1 {{std::casting forbids a C-style cast that casts away constness from 'const int *' to 'char *'}}

  // A suppression on a compound statement covers everything in it, including
  // lambdas.
  [[profiles::suppress(std::casting)]] {
    (void)reinterpret_cast<char *>(ip);
    auto L = [&] { return reinterpret_cast<char *>(ip); };
    (void)L;
  }

  // The suppression only covers its own statement.
  (void)reinterpret_cast<char *>(ip);
  // expected-error@-1 {{std::casting forbids reinterpret_cast from 'int *' to 'char *'}}
}

void declarations(int *ip) {
  [[profiles::suppress(std::casting)]] char *cp = reinterpret_cast<char *>(ip);
  (void)cp;
}

[[profiles::suppress(std::casting)]] char *suppressed_function(int *ip) {
  return reinterpret_cast<char *>(ip);
}

[[profiles::suppress(std::casting)]] char *suppressed_variable =
    reinterpret_cast<char *>(static_cast<int *>(nullptr));

char *unsuppressed_variable =
    reinterpret_cast<char *>(static_cast<int *>(nullptr));
// expected-error@-1 {{std::casting forbids reinterpret_cast from 'int *' to 'char *'}}

namespace [[profiles::suppress(std::casting)]] suppressed_namespace {
char *f(int *ip) { return reinterpret_cast<char *>(ip); }
} // namespace suppressed_namespace

struct [[profiles::suppress(std::casting)]] SuppressedClass {
  char *f(int *ip) { return reinterpret_cast<char *>(ip); }
};

template <typename T> char *suppressed_template(T *p) {
  [[profiles::suppress(std::casting)]] return reinterpret_cast<char *>(p);
}
char *use(int *ip) { return suppressed_template(ip); }

void diagnostics() {
  [[profiles::suppress(std::casting, rule: bogus)]] (void)0;
  // expected-warning@-1 {{unknown rule 'bogus' of profile 'std::casting'; the suppression has no effect}}
  [[profiles::suppress(acme::thing)]] (void)0;
  // expected-warning@-1 {{unknown profile 'acme::thing'}}
  [[profiles::suppress(std::casting, justification: unquoted)]] (void)0;
  // expected-error@-1 {{argument 'justification' of 'profiles::suppress' attribute must be a string literal}}
  [[profiles::suppress(std::casting, colour: red)]] (void)0;
  // expected-warning@-1 {{unknown argument 'colour' in 'profiles::suppress' attribute ignored}}
}
