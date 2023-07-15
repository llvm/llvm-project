// RUN: %clang_cc1 -fexperimental-new-constant-interpreter %s -verify
// RUN: %clang_cc1 -verify=ref %s -Wno-constant-evaluated
// RUN: %clang_cc1 -std=c++20 -fexperimental-new-constant-interpreter %s -verify
// RUN: %clang_cc1 -std=c++20 -verify=ref %s -Wno-constant-evaluated


namespace strcmp {
  constexpr char kFoobar[6] = {'f','o','o','b','a','r'};
  constexpr char kFoobazfoobar[12] = {'f','o','o','b','a','z','f','o','o','b','a','r'};

  static_assert(__builtin_strcmp("", "") == 0, "");
  static_assert(__builtin_strcmp("abab", "abab") == 0, "");
  static_assert(__builtin_strcmp("abab", "abba") == -1, "");
  static_assert(__builtin_strcmp("abab", "abaa") == 1, "");
  static_assert(__builtin_strcmp("ababa", "abab") == 1, "");
  static_assert(__builtin_strcmp("abab", "ababa") == -1, "");
  static_assert(__builtin_strcmp("a\203", "a") == 1, "");
  static_assert(__builtin_strcmp("a\203", "a\003") == 1, "");
  static_assert(__builtin_strcmp("abab\0banana", "abab") == 0, "");
  static_assert(__builtin_strcmp("abab", "abab\0banana") == 0, "");
  static_assert(__builtin_strcmp("abab\0banana", "abab\0canada") == 0, "");
  static_assert(__builtin_strcmp(0, "abab") == 0, ""); // expected-error {{not an integral constant}} \
                                                       // expected-note {{dereferenced null}} \
                                                       // expected-note {{in call to}} \
                                                       // ref-error {{not an integral constant}} \
                                                       // ref-note {{dereferenced null}}
  static_assert(__builtin_strcmp("abab", 0) == 0, ""); // expected-error {{not an integral constant}} \
                                                       // expected-note {{dereferenced null}} \
                                                       // expected-note {{in call to}} \
                                                       // ref-error {{not an integral constant}} \
                                                       // ref-note {{dereferenced null}}

  static_assert(__builtin_strcmp(kFoobar, kFoobazfoobar) == -1, "");
  static_assert(__builtin_strcmp(kFoobar, kFoobazfoobar + 6) == 0, ""); // expected-error {{not an integral constant}} \
                                                                        // expected-note {{dereferenced one-past-the-end}} \
                                                                        // expected-note {{in call to}} \
                                                                        // ref-error {{not an integral constant}} \
                                                                        // ref-note {{dereferenced one-past-the-end}}
}

namespace nan {
  constexpr double NaN1 = __builtin_nan("");

  /// The current interpreter does not accept this, but it should.
  constexpr float NaN2 = __builtin_nans([](){return "0xAE98";}()); // ref-error {{must be initialized by a constant expression}}

  constexpr double NaN3 = __builtin_nan("foo"); // expected-error {{must be initialized by a constant expression}} \
                                                // ref-error {{must be initialized by a constant expression}}
  constexpr float NaN4 = __builtin_nanf("");
  //constexpr long double NaN5 = __builtin_nanf128("");

  /// FIXME: This should be accepted by the current interpreter as well.
  constexpr char f[] = {'0', 'x', 'A', 'E', '\0'};
  constexpr double NaN6 = __builtin_nan(f); // ref-error {{must be initialized by a constant expression}}

  /// FIXME: Current interpreter misses diagnostics.
  constexpr char f2[] = {'0', 'x', 'A', 'E'}; /// No trailing 0 byte.
  constexpr double NaN7 = __builtin_nan(f2); // ref-error {{must be initialized by a constant expression}} \
                                             // expected-error {{must be initialized by a constant expression}} \
                                             // expected-note {{read of dereferenced one-past-the-end pointer}} \
                                             // expected-note {{in call to}}
}

namespace fmin {
  constexpr float f1 = __builtin_fmin(1.0, 2.0f);
  static_assert(f1 == 1.0f, "");

  constexpr float min = __builtin_fmin(__builtin_nan(""), 1);
  static_assert(min == 1, "");
  constexpr float min2 = __builtin_fmin(1, __builtin_nan(""));
  static_assert(min2 == 1, "");
  constexpr float min3 = __builtin_fmin(__builtin_inf(), __builtin_nan(""));
  static_assert(min3 == __builtin_inf(), "");
}

namespace inf {
  static_assert(__builtin_isinf(__builtin_inf()), "");
  static_assert(!__builtin_isinf(1.0), "");

  static_assert(__builtin_isfinite(1.0), "");
  static_assert(!__builtin_isfinite(__builtin_inf()), "");

  static_assert(__builtin_isnormal(1.0), "");
  static_assert(!__builtin_isnormal(__builtin_inf()), "");
}
