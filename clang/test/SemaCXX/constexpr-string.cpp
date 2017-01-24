// RUN: %clang_cc1 %s -std=c++1z -fsyntax-only -verify -pedantic
// RUN: %clang_cc1 %s -std=c++1z -fsyntax-only -verify -pedantic -fno-signed-char
// RUN: %clang_cc1 %s -std=c++1z -fsyntax-only -verify -pedantic -fno-wchar -Dwchar_t=__WCHAR_TYPE__

# 6 "/usr/include/string.h" 1 3 4
extern "C" {
  typedef decltype(sizeof(int)) size_t;

  extern size_t strlen(const char *p);

  extern int strcmp(const char *s1, const char *s2);
  extern int strncmp(const char *s1, const char *s2, size_t n);
  extern int memcmp(const void *s1, const void *s2, size_t n);

  extern char *strchr(const char *s, int c);
  extern void *memchr(const void *s, int c, size_t n);
}
# 19 "SemaCXX/constexpr-string.cpp" 2

# 21 "/usr/include/wchar.h" 1 3 4
extern "C" {
  extern size_t wcslen(const wchar_t *p);

  extern int wcscmp(const wchar_t *s1, const wchar_t *s2);
  extern int wcsncmp(const wchar_t *s1, const wchar_t *s2, size_t n);
  extern int wmemcmp(const wchar_t *s1, const wchar_t *s2, size_t n);

  extern wchar_t *wcschr(const wchar_t *s, wchar_t c);
  extern wchar_t *wmemchr(const wchar_t *s, wchar_t c, size_t n);
}

# 33 "SemaCXX/constexpr-string.cpp" 2
namespace Strlen {
  constexpr int n = __builtin_strlen("hello"); // ok
  static_assert(n == 5);
  constexpr int wn = __builtin_wcslen(L"hello"); // ok
  static_assert(wn == 5);
  constexpr int m = strlen("hello"); // expected-error {{constant expression}} expected-note {{non-constexpr function 'strlen' cannot be used in a constant expression}}
  constexpr int wm = wcslen(L"hello"); // expected-error {{constant expression}} expected-note {{non-constexpr function 'wcslen' cannot be used in a constant expression}}

  // Make sure we can evaluate a call to strlen.
  int arr[3]; // expected-note 2{{here}}
  int k = arr[strlen("hello")]; // expected-warning {{array index 5}}
  int wk = arr[wcslen(L"hello")]; // expected-warning {{array index 5}}
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

namespace WcscmpEtc {
  constexpr wchar_t kFoobar[6] = {L'f',L'o',L'o',L'b',L'a',L'r'};
  constexpr wchar_t kFoobazfoobar[12] = {L'f',L'o',L'o',L'b',L'a',L'z',L'f',L'o',L'o',L'b',L'a',L'r'};

  static_assert(__builtin_wcscmp(L"abab", L"abab") == 0);
  static_assert(__builtin_wcscmp(L"abab", L"abba") == -1);
  static_assert(__builtin_wcscmp(L"abab", L"abaa") == 1);
  static_assert(__builtin_wcscmp(L"ababa", L"abab") == 1);
  static_assert(__builtin_wcscmp(L"abab", L"ababa") == -1);
  static_assert(__builtin_wcscmp(L"abab\0banana", L"abab") == 0);
  static_assert(__builtin_wcscmp(L"abab", L"abab\0banana") == 0);
  static_assert(__builtin_wcscmp(L"abab\0banana", L"abab\0canada") == 0);
  static_assert(__builtin_wcscmp(0, L"abab") == 0); // expected-error {{not an integral constant}} expected-note {{dereferenced null}}
  static_assert(__builtin_wcscmp(L"abab", 0) == 0); // expected-error {{not an integral constant}} expected-note {{dereferenced null}}

  static_assert(__builtin_wcscmp(kFoobar, kFoobazfoobar) == -1); // FIXME: Should we reject this?
  static_assert(__builtin_wcscmp(kFoobar, kFoobazfoobar + 6) == 0); // expected-error {{not an integral constant}} expected-note {{dereferenced one-past-the-end}}

  static_assert(__builtin_wcsncmp(L"abaa", L"abba", 5) == -1);
  static_assert(__builtin_wcsncmp(L"abaa", L"abba", 4) == -1);
  static_assert(__builtin_wcsncmp(L"abaa", L"abba", 3) == -1);
  static_assert(__builtin_wcsncmp(L"abaa", L"abba", 2) == 0);
  static_assert(__builtin_wcsncmp(L"abaa", L"abba", 1) == 0);
  static_assert(__builtin_wcsncmp(L"abaa", L"abba", 0) == 0);
  static_assert(__builtin_wcsncmp(0, 0, 0) == 0);
  static_assert(__builtin_wcsncmp(L"abab\0banana", L"abab\0canada", 100) == 0);

  static_assert(__builtin_wcsncmp(kFoobar, kFoobazfoobar, 6) == -1);
  static_assert(__builtin_wcsncmp(kFoobar, kFoobazfoobar, 7) == -1); // FIXME: Should we reject this?
  static_assert(__builtin_wcsncmp(kFoobar, kFoobazfoobar + 6, 6) == 0);
  static_assert(__builtin_wcsncmp(kFoobar, kFoobazfoobar + 6, 7) == 0); // expected-error {{not an integral constant}} expected-note {{dereferenced one-past-the-end}}

  static_assert(__builtin_wmemcmp(L"abaa", L"abba", 3) == -1);
  static_assert(__builtin_wmemcmp(L"abaa", L"abba", 2) == 0);
  static_assert(__builtin_wmemcmp(0, 0, 0) == 0);
  static_assert(__builtin_wmemcmp(L"abab\0banana", L"abab\0banana", 100) == 0); // expected-error {{not an integral constant}} expected-note {{dereferenced one-past-the-end}}
  static_assert(__builtin_wmemcmp(L"abab\0banana", L"abab\0canada", 100) == -1); // FIXME: Should we reject this?
  static_assert(__builtin_wmemcmp(L"abab\0banana", L"abab\0canada", 7) == -1);
  static_assert(__builtin_wmemcmp(L"abab\0banana", L"abab\0canada", 6) == -1);
  static_assert(__builtin_wmemcmp(L"abab\0banana", L"abab\0canada", 5) == 0);

  constexpr int a = wcscmp(L"hello", L"world"); // expected-error {{constant expression}} expected-note {{non-constexpr function 'wcscmp' cannot be used in a constant expression}}
  constexpr int b = wcsncmp(L"hello", L"world", 3); // expected-error {{constant expression}} expected-note {{non-constexpr function 'wcsncmp' cannot be used in a constant expression}}
  constexpr int c = wmemcmp(L"hello", L"world", 3); // expected-error {{constant expression}} expected-note {{non-constexpr function 'wmemcmp' cannot be used in a constant expression}}
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

  static_assert(__builtin_char_memchr(kStr, 'a', 0) == nullptr);
  static_assert(__builtin_char_memchr(kStr, 'a', 1) == kStr);
  static_assert(__builtin_char_memchr(kStr, '\0', 5) == nullptr);
  static_assert(__builtin_char_memchr(kStr, '\0', 6) == kStr + 5);
  static_assert(__builtin_char_memchr(kStr, '\xff', 8) == kStr + 4);
  static_assert(__builtin_char_memchr(kStr, '\xff' + 256, 8) == kStr + 4);
  static_assert(__builtin_char_memchr(kStr, '\xff' - 256, 8) == kStr + 4);
  static_assert(__builtin_char_memchr(kFoo, 'x', 3) == nullptr);
  static_assert(__builtin_char_memchr(kFoo, 'x', 4) == nullptr); // expected-error {{not an integral constant}} expected-note {{dereferenced one-past-the-end}}
  static_assert(__builtin_char_memchr(nullptr, 'x', 3) == nullptr); // expected-error {{not an integral constant}} expected-note {{dereferenced null}}
  static_assert(__builtin_char_memchr(nullptr, 'x', 0) == nullptr); // FIXME: Should we reject this?

  static_assert(*__builtin_char_memchr(kStr, '\xff', 8) == '\xff');
  constexpr bool char_memchr_mutable() {
    char buffer[] = "mutable";
    *__builtin_char_memchr(buffer, 't', 8) = 'r';
    *__builtin_char_memchr(buffer, 'm', 8) = 'd';
    return __builtin_strcmp(buffer, "durable") == 0;
  }
  static_assert(char_memchr_mutable());

  constexpr bool a = !strchr("hello", 'h'); // expected-error {{constant expression}} expected-note {{non-constexpr function 'strchr' cannot be used in a constant expression}}
  constexpr bool b = !memchr("hello", 'h', 3); // expected-error {{constant expression}} expected-note {{non-constexpr function 'memchr' cannot be used in a constant expression}}
}

namespace WcschrEtc {
  constexpr const wchar_t *kStr = L"abca\xffff\0dL";
  constexpr wchar_t kFoo[] = {L'f', L'o', L'o'};
  static_assert(__builtin_wcschr(kStr, L'a') == kStr);
  static_assert(__builtin_wcschr(kStr, L'b') == kStr + 1);
  static_assert(__builtin_wcschr(kStr, L'c') == kStr + 2);
  static_assert(__builtin_wcschr(kStr, L'd') == nullptr);
  static_assert(__builtin_wcschr(kStr, L'e') == nullptr);
  static_assert(__builtin_wcschr(kStr, L'\0') == kStr + 5);
  static_assert(__builtin_wcschr(kStr, L'a' + 256) == nullptr);
  static_assert(__builtin_wcschr(kStr, L'a' - 256) == nullptr);
  static_assert(__builtin_wcschr(kStr, L'\xffff') == kStr + 4);
  static_assert(__builtin_wcschr(kFoo, L'o') == kFoo + 1);
  static_assert(__builtin_wcschr(kFoo, L'x') == nullptr); // expected-error {{not an integral constant}} expected-note {{dereferenced one-past-the-end}}
  static_assert(__builtin_wcschr(nullptr, L'x') == nullptr); // expected-error {{not an integral constant}} expected-note {{dereferenced null}}

  static_assert(__builtin_wmemchr(kStr, L'a', 0) == nullptr);
  static_assert(__builtin_wmemchr(kStr, L'a', 1) == kStr);
  static_assert(__builtin_wmemchr(kStr, L'\0', 5) == nullptr);
  static_assert(__builtin_wmemchr(kStr, L'\0', 6) == kStr + 5);
  static_assert(__builtin_wmemchr(kStr, L'\xffff', 8) == kStr + 4);
  static_assert(__builtin_wmemchr(kFoo, L'x', 3) == nullptr);
  static_assert(__builtin_wmemchr(kFoo, L'x', 4) == nullptr); // expected-error {{not an integral constant}} expected-note {{dereferenced one-past-the-end}}
  static_assert(__builtin_wmemchr(nullptr, L'x', 3) == nullptr); // expected-error {{not an integral constant}} expected-note {{dereferenced null}}
  static_assert(__builtin_wmemchr(nullptr, L'x', 0) == nullptr); // FIXME: Should we reject this?

  constexpr bool a = !wcschr(L"hello", L'h'); // expected-error {{constant expression}} expected-note {{non-constexpr function 'wcschr' cannot be used in a constant expression}}
  constexpr bool b = !wmemchr(L"hello", L'h', 3); // expected-error {{constant expression}} expected-note {{non-constexpr function 'wmemchr' cannot be used in a constant expression}}
}
