// RUN: %clang_cc1 %s -std=c++1z -fsyntax-only -verify -pedantic
// RUN: %clang_cc1 %s -std=c++1z -fsyntax-only -verify -pedantic -fno-signed-char

# 4 "/usr/include/string.h" 1 3 4
extern "C" {
  typedef decltype(sizeof(int)) size_t;

  extern size_t strlen(const char *p);

  extern int strcmp(const char *s1, const char *s2);
  extern int strncmp(const char *s1, const char *s2, size_t n);
  extern int memcmp(const char *s1, const char *s2, size_t n); // expected-note {{here}}

  extern char *strchr(const char *s, int c);
  extern void *memchr(const void *s, int c, size_t n);
}

# 19 "SemaCXX/constexpr-string.cpp" 2
namespace Strlen {
  constexpr int n = __builtin_strlen("hello"); // ok
  constexpr int m = strlen("hello"); // expected-error {{constant expression}} expected-note {{non-constexpr function 'strlen' cannot be used in a constant expression}}

  // Make sure we can evaluate a call to strlen.
  int arr[3]; // expected-note {{here}}
  int k = arr[strlen("hello")]; // expected-warning {{array index 5}}
}

namespace StrcmpEtc {
  constexpr char kFoobar[6] = {'f','o','o','b','a','r'};
  constexpr char kFoobazfoobar[12] = {'f','o','o','b','a','z','f','o','o','b','a','r'};

  static_assert(__builtin_strcmp("abab", "abab") == 0);
  static_assert(__builtin_strcmp("abab", "abba") == -1);
  static_assert(__builtin_strcmp("abab", "abaa") == 1);
  static_assert(__builtin_strcmp("ababa", "abab") == 1);
  static_assert(__builtin_strcmp("abab", "ababa") == -1);
  static_assert(__builtin_strcmp("abab\0banana", "abab") == 0);
  static_assert(__builtin_strcmp("abab", "abab\0banana") == 0);
  static_assert(__builtin_strcmp("abab\0banana", "abab\0canada") == 0);
  static_assert(__builtin_strcmp(0, "abab") == 0); // expected-error {{not an integral constant}} expected-note {{dereferenced null}}
  static_assert(__builtin_strcmp("abab", 0) == 0); // expected-error {{not an integral constant}} expected-note {{dereferenced null}}

  static_assert(__builtin_strcmp(kFoobar, kFoobazfoobar) == -1); // FIXME: Should we reject this?
  static_assert(__builtin_strcmp(kFoobar, kFoobazfoobar + 6) == 0); // expected-error {{not an integral constant}} expected-note {{dereferenced one-past-the-end}}

  static_assert(__builtin_strncmp("abaa", "abba", 5) == -1);
  static_assert(__builtin_strncmp("abaa", "abba", 4) == -1);
  static_assert(__builtin_strncmp("abaa", "abba", 3) == -1);
  static_assert(__builtin_strncmp("abaa", "abba", 2) == 0);
  static_assert(__builtin_strncmp("abaa", "abba", 1) == 0);
  static_assert(__builtin_strncmp("abaa", "abba", 0) == 0);
  static_assert(__builtin_strncmp(0, 0, 0) == 0);
  static_assert(__builtin_strncmp("abab\0banana", "abab\0canada", 100) == 0);

  static_assert(__builtin_strncmp(kFoobar, kFoobazfoobar, 6) == -1);
  static_assert(__builtin_strncmp(kFoobar, kFoobazfoobar, 7) == -1); // FIXME: Should we reject this?
  static_assert(__builtin_strncmp(kFoobar, kFoobazfoobar + 6, 6) == 0);
  static_assert(__builtin_strncmp(kFoobar, kFoobazfoobar + 6, 7) == 0); // expected-error {{not an integral constant}} expected-note {{dereferenced one-past-the-end}}

  static_assert(__builtin_memcmp("abaa", "abba", 3) == -1);
  static_assert(__builtin_memcmp("abaa", "abba", 2) == 0);
  static_assert(__builtin_memcmp(0, 0, 0) == 0);
  static_assert(__builtin_memcmp("abab\0banana", "abab\0banana", 100) == 0); // expected-error {{not an integral constant}} expected-note {{dereferenced one-past-the-end}}
  static_assert(__builtin_memcmp("abab\0banana", "abab\0canada", 100) == -1); // FIXME: Should we reject this?
  static_assert(__builtin_memcmp("abab\0banana", "abab\0canada", 7) == -1);
  static_assert(__builtin_memcmp("abab\0banana", "abab\0canada", 6) == -1);
  static_assert(__builtin_memcmp("abab\0banana", "abab\0canada", 5) == 0);

  constexpr int a = strcmp("hello", "world"); // expected-error {{constant expression}} expected-note {{non-constexpr function 'strcmp' cannot be used in a constant expression}}
  constexpr int b = strncmp("hello", "world", 3); // expected-error {{constant expression}} expected-note {{non-constexpr function 'strncmp' cannot be used in a constant expression}}
  constexpr int c = memcmp("hello", "world", 3); // expected-error {{constant expression}} expected-note {{non-constexpr function 'memcmp' cannot be used in a constant expression}}
}

namespace StrchrEtc {
  constexpr const char *kStr = "abca\xff\0d";
  constexpr char kFoo[] = {'f', 'o', 'o'};
  static_assert(__builtin_strchr(kStr, 'a') == kStr);
  static_assert(__builtin_strchr(kStr, 'b') == kStr + 1);
  static_assert(__builtin_strchr(kStr, 'c') == kStr + 2);
  static_assert(__builtin_strchr(kStr, 'd') == nullptr);
  static_assert(__builtin_strchr(kStr, 'e') == nullptr);
  static_assert(__builtin_strchr(kStr, '\0') == kStr + 5);
  static_assert(__builtin_strchr(kStr, 'a' + 256) == nullptr);
  static_assert(__builtin_strchr(kStr, 'a' - 256) == nullptr);
  static_assert(__builtin_strchr(kStr, '\xff') == kStr + 4);
  static_assert(__builtin_strchr(kStr, '\xff' + 256) == nullptr);
  static_assert(__builtin_strchr(kStr, '\xff' - 256) == nullptr);
  static_assert(__builtin_strchr(kFoo, 'o') == kFoo + 1);
  static_assert(__builtin_strchr(kFoo, 'x') == nullptr); // expected-error {{not an integral constant}} expected-note {{dereferenced one-past-the-end}}
  static_assert(__builtin_strchr(nullptr, 'x') == nullptr); // expected-error {{not an integral constant}} expected-note {{dereferenced null}}

  static_assert(__builtin_memchr(kStr, 'a', 0) == nullptr);
  static_assert(__builtin_memchr(kStr, 'a', 1) == kStr);
  static_assert(__builtin_memchr(kStr, '\0', 5) == nullptr);
  static_assert(__builtin_memchr(kStr, '\0', 6) == kStr + 5);
  static_assert(__builtin_memchr(kStr, '\xff', 8) == kStr + 4);
  static_assert(__builtin_memchr(kStr, '\xff' + 256, 8) == kStr + 4);
  static_assert(__builtin_memchr(kStr, '\xff' - 256, 8) == kStr + 4);
  static_assert(__builtin_memchr(kFoo, 'x', 3) == nullptr);
  static_assert(__builtin_memchr(kFoo, 'x', 4) == nullptr); // expected-error {{not an integral constant}} expected-note {{dereferenced one-past-the-end}}
  static_assert(__builtin_memchr(nullptr, 'x', 3) == nullptr); // expected-error {{not an integral constant}} expected-note {{dereferenced null}}
  static_assert(__builtin_memchr(nullptr, 'x', 0) == nullptr); // FIXME: Should we reject this?

  constexpr bool a = !strchr("hello", 'h'); // expected-error {{constant expression}} expected-note {{non-constexpr function 'strchr' cannot be used in a constant expression}}
  constexpr bool b = !memchr("hello", 'h', 3); // expected-error {{constant expression}} expected-note {{non-constexpr function 'memchr' cannot be used in a constant expression}}
}
