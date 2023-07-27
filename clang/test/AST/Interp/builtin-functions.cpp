// RUN: %clang_cc1 -fexperimental-new-constant-interpreter %s -verify
// RUN: %clang_cc1 -verify=ref %s -Wno-constant-evaluated

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
