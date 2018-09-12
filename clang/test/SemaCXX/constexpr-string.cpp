// RUN: %clang_cc1 %s -triple x86_64-linux-gnu -std=c++1z -fsyntax-only -verify -pedantic
// RUN: %clang_cc1 %s -triple x86_64-linux-gnu -std=c++1z -fsyntax-only -verify -pedantic -fno-signed-char
// RUN: %clang_cc1 %s -triple x86_64-linux-gnu -std=c++1z -fsyntax-only -verify -pedantic -fno-wchar -Dwchar_t=__WCHAR_TYPE__

# 6 "/usr/include/string.h" 1 3 4
extern "C" {
  typedef decltype(sizeof(int)) size_t;

  extern size_t strlen(const char *p);

  extern int strcmp(const char *s1, const char *s2);
  extern int strncmp(const char *s1, const char *s2, size_t n);
  extern int memcmp(const void *s1, const void *s2, size_t n);

  extern char *strchr(const char *s, int c);
  extern void *memchr(const void *s, int c, size_t n);

  extern void *memcpy(void *d, const void *s, size_t n);
  extern void *memmove(void *d, const void *s, size_t n);
}
# 22 "SemaCXX/constexpr-string.cpp" 2

# 24 "/usr/include/wchar.h" 1 3 4
extern "C" {
  extern size_t wcslen(const wchar_t *p);

  extern int wcscmp(const wchar_t *s1, const wchar_t *s2);
  extern int wcsncmp(const wchar_t *s1, const wchar_t *s2, size_t n);
  extern int wmemcmp(const wchar_t *s1, const wchar_t *s2, size_t n);

  extern wchar_t *wcschr(const wchar_t *s, wchar_t c);
  extern wchar_t *wmemchr(const wchar_t *s, wchar_t c, size_t n);

  extern wchar_t *wmemcpy(wchar_t *d, const wchar_t *s, size_t n);
  extern wchar_t *wmemmove(wchar_t *d, const wchar_t *s, size_t n);
}

# 39 "SemaCXX/constexpr-string.cpp" 2
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
  static_assert(__builtin_strcmp("a\203", "a") == 1);
  static_assert(__builtin_strcmp("a\203", "a\003") == 1);
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
  static_assert(__builtin_memcmp("a\203", "a", 2) == 1);
  static_assert(__builtin_memcmp("a\203", "a\003", 2) == 1);
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
#if __WCHAR_WIDTH__ == 32
  static_assert(__builtin_wcscmp(L"a\x83838383", L"a") == (wchar_t)-1U >> 31);
#endif
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
#if __WCHAR_WIDTH__ == 32
  static_assert(__builtin_wcsncmp(L"a\x83838383", L"aa", 2) ==
                (wchar_t)-1U >> 31);
#endif

  static_assert(__builtin_wcsncmp(kFoobar, kFoobazfoobar, 6) == -1);
  static_assert(__builtin_wcsncmp(kFoobar, kFoobazfoobar, 7) == -1); // FIXME: Should we reject this?
  static_assert(__builtin_wcsncmp(kFoobar, kFoobazfoobar + 6, 6) == 0);
  static_assert(__builtin_wcsncmp(kFoobar, kFoobazfoobar + 6, 7) == 0); // expected-error {{not an integral constant}} expected-note {{dereferenced one-past-the-end}}

  static_assert(__builtin_wmemcmp(L"abaa", L"abba", 3) == -1);
  static_assert(__builtin_wmemcmp(L"abaa", L"abba", 2) == 0);
  static_assert(__builtin_wmemcmp(0, 0, 0) == 0);
#if __WCHAR_WIDTH__ == 32
  static_assert(__builtin_wmemcmp(L"a\x83838383", L"aa", 2) ==
                (wchar_t)-1U >> 31);
#endif
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

namespace MemcpyEtc {
  template<typename T>
  constexpr T result(T (&arr)[4]) {
    return arr[0] * 1000 + arr[1] * 100 + arr[2] * 10 + arr[3];
  }

  constexpr int test_memcpy(int a, int b, int n) {
    int arr[4] = {1, 2, 3, 4};
    __builtin_memcpy(arr + a, arr + b, n);
    // expected-note@-1 2{{overlapping memory regions}}
    // expected-note@-2 {{size to copy (1) is not a multiple of size of element type 'int'}}
    // expected-note@-3 {{source is not a contiguous array of at least 2 elements of type 'int'}}
    // expected-note@-4 {{destination is not a contiguous array of at least 3 elements of type 'int'}}
    return result(arr);
  }
  constexpr int test_memmove(int a, int b, int n) {
    int arr[4] = {1, 2, 3, 4};
    __builtin_memmove(arr + a, arr + b, n);
    // expected-note@-1 {{size to copy (1) is not a multiple of size of element type 'int'}}
    // expected-note@-2 {{source is not a contiguous array of at least 2 elements of type 'int'}}
    // expected-note@-3 {{destination is not a contiguous array of at least 3 elements of type 'int'}}
    return result(arr);
  }
  constexpr int test_wmemcpy(int a, int b, int n) {
    wchar_t arr[4] = {1, 2, 3, 4};
    __builtin_wmemcpy(arr + a, arr + b, n);
    // expected-note@-1 2{{overlapping memory regions}}
    // expected-note-re@-2 {{source is not a contiguous array of at least 2 elements of type '{{wchar_t|int}}'}}
    // expected-note-re@-3 {{destination is not a contiguous array of at least 3 elements of type '{{wchar_t|int}}'}}
    return result(arr);
  }
  constexpr int test_wmemmove(int a, int b, int n) {
    wchar_t arr[4] = {1, 2, 3, 4};
    __builtin_wmemmove(arr + a, arr + b, n);
    // expected-note-re@-1 {{source is not a contiguous array of at least 2 elements of type '{{wchar_t|int}}'}}
    // expected-note-re@-2 {{destination is not a contiguous array of at least 3 elements of type '{{wchar_t|int}}'}}
    return result(arr);
  }

  static_assert(test_memcpy(1, 2, 4) == 1334);
  static_assert(test_memcpy(2, 1, 4) == 1224);
  static_assert(test_memcpy(0, 1, 8) == 2334); // expected-error {{constant}} expected-note {{in call}}
  static_assert(test_memcpy(1, 0, 8) == 1124); // expected-error {{constant}} expected-note {{in call}}
  static_assert(test_memcpy(1, 2, 1) == 1334); // expected-error {{constant}} expected-note {{in call}}
  static_assert(test_memcpy(0, 3, 4) == 4234);
  static_assert(test_memcpy(0, 3, 8) == 4234); // expected-error {{constant}} expected-note {{in call}}
  static_assert(test_memcpy(2, 0, 12) == 4234); // expected-error {{constant}} expected-note {{in call}}

  static_assert(test_memmove(1, 2, 4) == 1334);
  static_assert(test_memmove(2, 1, 4) == 1224);
  static_assert(test_memmove(0, 1, 8) == 2334);
  static_assert(test_memmove(1, 0, 8) == 1124);
  static_assert(test_memmove(1, 2, 1) == 1334); // expected-error {{constant}} expected-note {{in call}}
  static_assert(test_memmove(0, 3, 4) == 4234);
  static_assert(test_memmove(0, 3, 8) == 4234); // expected-error {{constant}} expected-note {{in call}}
  static_assert(test_memmove(2, 0, 12) == 4234); // expected-error {{constant}} expected-note {{in call}}

  static_assert(test_wmemcpy(1, 2, 1) == 1334);
  static_assert(test_wmemcpy(2, 1, 1) == 1224);
  static_assert(test_wmemcpy(0, 1, 2) == 2334); // expected-error {{constant}} expected-note {{in call}}
  static_assert(test_wmemcpy(1, 0, 2) == 1124); // expected-error {{constant}} expected-note {{in call}}
  static_assert(test_wmemcpy(1, 2, 1) == 1334);
  static_assert(test_wmemcpy(0, 3, 1) == 4234);
  static_assert(test_wmemcpy(0, 3, 2) == 4234); // expected-error {{constant}} expected-note {{in call}}
  static_assert(test_wmemcpy(2, 0, 3) == 4234); // expected-error {{constant}} expected-note {{in call}}

  static_assert(test_wmemmove(1, 2, 1) == 1334);
  static_assert(test_wmemmove(2, 1, 1) == 1224);
  static_assert(test_wmemmove(0, 1, 2) == 2334);
  static_assert(test_wmemmove(1, 0, 2) == 1124);
  static_assert(test_wmemmove(1, 2, 1) == 1334);
  static_assert(test_wmemmove(0, 3, 1) == 4234);
  static_assert(test_wmemmove(0, 3, 2) == 4234); // expected-error {{constant}} expected-note {{in call}}
  static_assert(test_wmemmove(2, 0, 3) == 4234); // expected-error {{constant}} expected-note {{in call}}

  // Copying is permitted for any trivially-copyable type.
  struct Trivial { char k; short s; constexpr bool ok() { return k == 3 && s == 4; } };
  constexpr bool test_trivial() {
    Trivial arr[3] = {{1, 2}, {3, 4}, {5, 6}};
    __builtin_memcpy(arr, arr+1, sizeof(Trivial));
    __builtin_memmove(arr+1, arr, 2 * sizeof(Trivial));
    return arr[0].ok() && arr[1].ok() && arr[2].ok();
  }
  static_assert(test_trivial());

  // But not for a non-trivially-copyable type.
  struct NonTrivial {
    constexpr NonTrivial() : n(0) {}
    constexpr NonTrivial(const NonTrivial &) : n(1) {}
    int n;
  };
  constexpr bool test_nontrivial_memcpy() { // expected-error {{never produces a constant}}
    NonTrivial arr[3] = {};
    __builtin_memcpy(arr, arr + 1, sizeof(NonTrivial)); // expected-note 2{{non-trivially-copyable}}
    return true;
  }
  static_assert(test_nontrivial_memcpy()); // expected-error {{constant}} expected-note {{in call}}
  constexpr bool test_nontrivial_memmove() { // expected-error {{never produces a constant}}
    NonTrivial arr[3] = {};
    __builtin_memcpy(arr, arr + 1, sizeof(NonTrivial)); // expected-note 2{{non-trivially-copyable}}
    return true;
  }
  static_assert(test_nontrivial_memmove()); // expected-error {{constant}} expected-note {{in call}}

  // Type puns via constant evaluated memcpy are not supported yet.
  constexpr float type_pun(const unsigned &n) {
    float f = 0.0f;
    __builtin_memcpy(&f, &n, 4); // expected-note {{cannot constant evaluate 'memcpy' from object of type 'const unsigned int' to object of type 'float'}}
    return f;
  }
  static_assert(type_pun(0x3f800000) == 1.0f); // expected-error {{constant}} expected-note {{in call}}

  // Make sure we're not confused by derived-to-base conversions.
  struct Base { int a; };
  struct Derived : Base { int b; };
  constexpr int test_derived_to_base(int n) {
    Derived arr[2] = {1, 2, 3, 4};
    Base *p = &arr[0];
    Base *q = &arr[1];
    __builtin_memcpy(p, q, sizeof(Base) * n); // expected-note {{source is not a contiguous array of at least 2 elements of type 'MemcpyEtc::Base'}}
    return arr[0].a * 1000 + arr[0].b * 100 + arr[1].a * 10 + arr[1].b;
  }
  static_assert(test_derived_to_base(0) == 1234);
  static_assert(test_derived_to_base(1) == 3234);
  // FIXME: We could consider making this work by stripping elements off both
  // designators until we have a long enough matching size, if both designators
  // point to the start of their respective final elements.
  static_assert(test_derived_to_base(2) == 3434); // expected-error {{constant}} expected-note {{in call}}
}
