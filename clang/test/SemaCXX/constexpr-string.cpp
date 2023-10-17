// RUN: %clang_cc1 %s -triple x86_64-linux-gnu -std=c++2a -fsyntax-only -verify -pedantic -Wno-vla-extension
// RUN: %clang_cc1 %s -triple x86_64-linux-gnu -std=gnu++2a -fsyntax-only -verify -pedantic -Wno-vla-extension -DGNUMODE
// RUN: %clang_cc1 %s -triple x86_64-linux-gnu -std=c++2a -fsyntax-only -verify -pedantic -Wno-vla-extension -fno-signed-char
// RUN: %clang_cc1 %s -triple x86_64-linux-gnu -std=c++2a -fsyntax-only -verify -pedantic -Wno-vla-extension -fno-wchar -DNO_PREDEFINED_WCHAR_T
// RUN: %clang_cc1 %s -triple armebv7-unknown-linux -std=c++2a -fsyntax-only -verify -pedantic -Wno-vla-extension
// RUN: %clang_cc1 %s -triple armebv7-unknown-linux -std=gnu++2a -fsyntax-only -verify -pedantic -Wno-vla-extension -DGNUMODE
// RUN: %clang_cc1 %s -triple armebv7-unknown-linux -std=c++2a -fsyntax-only -verify -pedantic -Wno-vla-extension -fno-signed-char
// RUN: %clang_cc1 %s -triple armebv7-unknown-linux -std=c++2a -fsyntax-only -verify -pedantic -Wno-vla-extension -fno-wchar -DNO_PREDEFINED_WCHAR_T

# 9 "/usr/include/string.h" 1 3 4  // expected-warning {{this style of line directive is a GNU extension}}
extern "C" {
  typedef decltype(sizeof(int)) size_t;

  extern size_t strlen(const char *p);

  extern int strcmp(const char *s1, const char *s2);
  extern int strncmp(const char *s1, const char *s2, size_t n);
  extern int memcmp(const void *s1, const void *s2, size_t n);

#ifdef GNUMODE
  extern int bcmp(const void *s1, const void *s2, size_t n);
#endif

  extern char *strchr(const char *s, int c);
  extern void *memchr(const void *s, int c, size_t n);

  extern void *memcpy(void *d, const void *s, size_t n);
  extern void *memmove(void *d, const void *s, size_t n);
}
# 25 "SemaCXX/constexpr-string.cpp" 2

# 27 "/usr/include/wchar.h" 1 3 4  // expected-warning {{this style of line directive is a GNU extension}}
extern "C" {
#if NO_PREDEFINED_WCHAR_T
  typedef decltype(L'0') wchar_t;
#endif
  extern size_t wcslen(const wchar_t *p);

  extern int wcscmp(const wchar_t *s1, const wchar_t *s2);
  extern int wcsncmp(const wchar_t *s1, const wchar_t *s2, size_t n);
  extern int wmemcmp(const wchar_t *s1, const wchar_t *s2, size_t n);

  extern wchar_t *wcschr(const wchar_t *s, wchar_t c);
  extern wchar_t *wmemchr(const wchar_t *s, wchar_t c, size_t n);

  extern wchar_t *wmemcpy(wchar_t *d, const wchar_t *s, size_t n);
  extern wchar_t *wmemmove(wchar_t *d, const wchar_t *s, size_t n);
}

# 51 "SemaCXX/constexpr-string.cpp" 2
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

  static_assert(__builtin_memcmp(u8"abaa", u8"abba", 3) == -1);
  static_assert(__builtin_memcmp(u8"abaa", u8"abba", 2) == 0);
  static_assert(__builtin_memcmp(u8"a\203", u8"a", 2) == 1);
  static_assert(__builtin_memcmp(u8"a\203", u8"a\003", 2) == 1);
  static_assert(__builtin_memcmp(0, 0, 0) == 0);
  static_assert(__builtin_memcmp(u8"abab\0banana", u8"abab\0banana", 100) == 0); // expected-error {{not an integral constant}} expected-note {{dereferenced one-past-the-end}}
  static_assert(__builtin_memcmp(u8"abab\0banana", u8"abab\0canada", 100) == -1); // FIXME: Should we reject this?
  static_assert(__builtin_memcmp(u8"abab\0banana", u8"abab\0canada", 7) == -1);
  static_assert(__builtin_memcmp(u8"abab\0banana", u8"abab\0canada", 6) == -1);
  static_assert(__builtin_memcmp(u8"abab\0banana", u8"abab\0canada", 5) == 0);

  static_assert(__builtin_memcmp(u8"\u1234", "\xE1\x88\xB4", 4) == 0);
  static_assert(__builtin_memcmp(u8"\u1234", "\xE1\x88\xB3", 4) == 1);

  static_assert(__builtin_bcmp("abaa", "abba", 3) != 0);
  static_assert(__builtin_bcmp("abaa", "abba", 2) == 0);
  static_assert(__builtin_bcmp("a\203", "a", 2) != 0);
  static_assert(__builtin_bcmp("a\203", "a\003", 2) != 0);
  static_assert(__builtin_bcmp(0, 0, 0) == 0);
  static_assert(__builtin_bcmp("abab\0banana", "abab\0banana", 100) == 0); // expected-error {{not an integral constant}} expected-note {{dereferenced one-past-the-end}}
  static_assert(__builtin_bcmp("abab\0banana", "abab\0canada", 100) != 0); // FIXME: Should we reject this?
  static_assert(__builtin_bcmp("abab\0banana", "abab\0canada", 7) != 0);
  static_assert(__builtin_bcmp("abab\0banana", "abab\0canada", 6) != 0);
  static_assert(__builtin_bcmp("abab\0banana", "abab\0canada", 5) == 0);

  extern struct Incomplete incomplete;
  static_assert(__builtin_memcmp(&incomplete, "", 0u) == 0);
  static_assert(__builtin_memcmp("", &incomplete, 0u) == 0);
  static_assert(__builtin_memcmp(&incomplete, "", 1u) == 42); // expected-error {{not an integral constant}} expected-note {{not supported}}
  static_assert(__builtin_memcmp("", &incomplete, 1u) == 42); // expected-error {{not an integral constant}} expected-note {{not supported}}

  static_assert(__builtin_bcmp(&incomplete, "", 0u) == 0);
  static_assert(__builtin_bcmp("", &incomplete, 0u) == 0);
  static_assert(__builtin_bcmp(&incomplete, "", 1u) == 42); // expected-error {{not an integral constant}} expected-note {{not supported}}
  static_assert(__builtin_bcmp("", &incomplete, 1u) == 42); // expected-error {{not an integral constant}} expected-note {{not supported}}

  constexpr unsigned char ku00fe00[] = {0x00, 0xfe, 0x00};
  constexpr unsigned char ku00feff[] = {0x00, 0xfe, 0xff};
  constexpr signed char ks00fe00[] = {0, -2, 0};
  constexpr signed char ks00feff[] = {0, -2, -1};
  static_assert(__builtin_memcmp(ku00feff, ks00fe00, 2) == 0);
  static_assert(__builtin_memcmp(ku00feff, ks00fe00, 99) == 1);
  static_assert(__builtin_memcmp(ku00fe00, ks00feff, 99) == -1);
  static_assert(__builtin_memcmp(ks00feff, ku00fe00, 2) == 0);
  static_assert(__builtin_memcmp(ks00feff, ku00fe00, 99) == 1);
  static_assert(__builtin_memcmp(ks00fe00, ku00feff, 99) == -1);
  static_assert(__builtin_memcmp(ks00fe00, ks00feff, 2) == 0);
  static_assert(__builtin_memcmp(ks00feff, ks00fe00, 99) == 1);
  static_assert(__builtin_memcmp(ks00fe00, ks00feff, 99) == -1);

  static_assert(__builtin_bcmp(ku00feff, ks00fe00, 2) == 0);
  static_assert(__builtin_bcmp(ku00feff, ks00fe00, 99) != 0);
  static_assert(__builtin_bcmp(ku00fe00, ks00feff, 99) != 0);
  static_assert(__builtin_bcmp(ks00feff, ku00fe00, 2) == 0);
  static_assert(__builtin_bcmp(ks00feff, ku00fe00, 99) != 0);
  static_assert(__builtin_bcmp(ks00fe00, ku00feff, 99) != 0);
  static_assert(__builtin_bcmp(ks00fe00, ks00feff, 2) == 0);
  static_assert(__builtin_bcmp(ks00feff, ks00fe00, 99) != 0);
  static_assert(__builtin_bcmp(ks00fe00, ks00feff, 99) != 0);

  struct Bool3Tuple { bool bb[3]; };
  constexpr Bool3Tuple kb000100 = {{false, true, false}};
  static_assert(sizeof(bool) != 1u || __builtin_memcmp(ks00fe00, kb000100.bb, 1) == 0); // expected-error {{constant}} expected-note {{not supported}}
  static_assert(sizeof(bool) != 1u || __builtin_memcmp(ks00fe00, kb000100.bb, 2) == 1); // expected-error {{constant}} expected-note {{not supported}}

  static_assert(sizeof(bool) != 1u || __builtin_bcmp(ks00fe00, kb000100.bb, 1) == 0); // expected-error {{constant}} expected-note {{not supported}}
  static_assert(sizeof(bool) != 1u || __builtin_bcmp(ks00fe00, kb000100.bb, 2) != 0); // expected-error {{constant}} expected-note {{not supported}}

  constexpr long ksl[] = {0, -1};
  constexpr unsigned int kui[] = {0, 0u - 1};
  constexpr unsigned long long kull[] = {0, 0ull - 1};
  constexpr const auto *kuSizeofLong(void) {
    if constexpr(sizeof(long) == sizeof(int)) {
      return kui;
    } else if constexpr(sizeof(long) == sizeof(long long)) {
      return kull;
    } else {
      return nullptr;
    }
  }
  static_assert(__builtin_memcmp(ksl, kuSizeofLong(), sizeof(long) - 1) == 0); // expected-error {{constant}} expected-note {{not supported}}
  static_assert(__builtin_memcmp(ksl, kuSizeofLong(), sizeof(long) + 0) == 0); // expected-error {{constant}} expected-note {{not supported}}
  static_assert(__builtin_memcmp(ksl, kuSizeofLong(), sizeof(long) + 1) == 0); // expected-error {{constant}} expected-note {{not supported}}
  static_assert(__builtin_memcmp(ksl, kuSizeofLong(), 2*sizeof(long) - 1) == 0); // expected-error {{constant}} expected-note {{not supported}}
  static_assert(__builtin_memcmp(ksl, kuSizeofLong(), 2*sizeof(long) + 0) == 0); // expected-error {{constant}} expected-note {{not supported}}
  static_assert(__builtin_memcmp(ksl, kuSizeofLong(), 2*sizeof(long) + 1) == 42); // expected-error {{constant}} expected-note {{not supported}}
  static_assert(__builtin_memcmp(ksl + 1, kuSizeofLong() + 1, sizeof(long) - 1) == 0); // expected-error {{constant}} expected-note {{not supported}}
  static_assert(__builtin_memcmp(ksl + 1, kuSizeofLong() + 1, sizeof(long) + 0) == 0); // expected-error {{constant}} expected-note {{not supported}}
  static_assert(__builtin_memcmp(ksl + 1, kuSizeofLong() + 1, sizeof(long) + 1) == 42); // expected-error {{constant}} expected-note {{not supported}}

  static_assert(__builtin_bcmp(ksl, kuSizeofLong(), sizeof(long) - 1) == 0); // expected-error {{constant}} expected-note {{not supported}}
  static_assert(__builtin_bcmp(ksl, kuSizeofLong(), sizeof(long) + 0) == 0); // expected-error {{constant}} expected-note {{not supported}}
  static_assert(__builtin_bcmp(ksl, kuSizeofLong(), sizeof(long) + 1) == 0); // expected-error {{constant}} expected-note {{not supported}}
  static_assert(__builtin_bcmp(ksl, kuSizeofLong(), 2*sizeof(long) - 1) == 0); // expected-error {{constant}} expected-note {{not supported}}
  static_assert(__builtin_bcmp(ksl, kuSizeofLong(), 2*sizeof(long) + 0) == 0); // expected-error {{constant}} expected-note {{not supported}}
  static_assert(__builtin_bcmp(ksl, kuSizeofLong(), 2*sizeof(long) + 1) == 42); // expected-error {{constant}} expected-note {{not supported}}
  static_assert(__builtin_bcmp(ksl + 1, kuSizeofLong() + 1, sizeof(long) - 1) == 0); // expected-error {{constant}} expected-note {{not supported}}
  static_assert(__builtin_bcmp(ksl + 1, kuSizeofLong() + 1, sizeof(long) + 0) == 0); // expected-error {{constant}} expected-note {{not supported}}
  static_assert(__builtin_bcmp(ksl + 1, kuSizeofLong() + 1, sizeof(long) + 1) == 42); // expected-error {{constant}} expected-note {{not supported}}

  constexpr int a = strcmp("hello", "world"); // expected-error {{constant expression}} expected-note {{non-constexpr function 'strcmp' cannot be used in a constant expression}}
  constexpr int b = strncmp("hello", "world", 3); // expected-error {{constant expression}} expected-note {{non-constexpr function 'strncmp' cannot be used in a constant expression}}
  constexpr int c = memcmp("hello", "world", 3); // expected-error {{constant expression}} expected-note {{non-constexpr function 'memcmp' cannot be used in a constant expression}}

#ifdef GNUMODE
  constexpr int d = bcmp("hello", "world", 3); // expected-error {{constant expression}} expected-note {{non-constexpr function 'bcmp' cannot be used in a constant expression}}
#endif
}

namespace MultibyteElementTests {
inline namespace Util {
#define STR2(X) #X
#define STR(X) STR2(X)
constexpr const char ByteOrderString[] = STR(__BYTE_ORDER__);
#undef STR
#undef STR2
constexpr bool LittleEndian{*ByteOrderString == '1'};

constexpr size_t GoodFoldArraySize = 42, BadFoldArraySize = 43;
struct NotBadFoldResult {};
template <size_t> struct FoldResult;
template <> struct FoldResult<GoodFoldArraySize> : NotBadFoldResult {};
template <typename T, size_t N>
FoldResult<N> *foldResultImpl(T (*ptrToConstantSizeArray)[N]);
struct NotFolded : NotBadFoldResult {};
NotFolded *foldResultImpl(bool anyPtr);
template <auto Value> struct MetaValue;
template <typename Callable, size_t N, auto ExpectedFoldResult>
auto foldResult(const Callable &, MetaValue<N> *,
                MetaValue<ExpectedFoldResult> *) {
  int (*maybeVLAPtr)[Callable{}(N) == ExpectedFoldResult
                         ? GoodFoldArraySize
                         : BadFoldArraySize] = 0;
  return foldResultImpl(maybeVLAPtr);
}
template <typename FoldResultKind, typename Callable, typename NWrap,
          typename ExpectedWrap>
constexpr bool checkFoldResult(const Callable &c, NWrap *n, ExpectedWrap *e) {
  decltype(static_cast<FoldResultKind *>(foldResult(c, n, e))) *chk{};
  return true;
}
template <size_t N> constexpr MetaValue<N> *withN() { return nullptr; }
template <auto Expected> constexpr MetaValue<Expected> *withExpected() {
  return nullptr;
}
} // namespace Util
} // namespace MultibyteElementTests

namespace MultibyteElementTests::Memcmp {
#ifdef __SIZEOF_INT128__
constexpr __int128 i128_ff_8_00_8 = -(__int128)1 - -1ull;
constexpr __int128 i128_00_16 = 0;
static_assert(checkFoldResult<NotBadFoldResult>(
    [](size_t n) constexpr {
      return __builtin_memcmp(&i128_ff_8_00_8, &i128_00_16, n);
    },
    withN<1u>(), withExpected<LittleEndian ? 0 : 1>()));
#endif

constexpr const signed char ByteOrderStringReduced[] = {
    ByteOrderString[0] - '0', ByteOrderString[1] - '0',
    ByteOrderString[2] - '0', ByteOrderString[3] - '0',
};
constexpr signed int i04030201 = 0x04030201;
constexpr unsigned int u04030201 = 0x04030201u;
static_assert(checkFoldResult<NotBadFoldResult>(
    [](size_t n) constexpr {
      return __builtin_memcmp(ByteOrderStringReduced, &i04030201, n);
    },
    withN<sizeof(int)>(), withExpected<0>()));
static_assert(checkFoldResult<NotBadFoldResult>(
    [](size_t n) constexpr {
      return __builtin_memcmp(&u04030201, ByteOrderStringReduced, n);
    },
    withN<sizeof(int)>(), withExpected<0>()));

constexpr unsigned int ui0000FEFF = 0x0000feffU;
constexpr unsigned short usFEFF = 0xfeffU;
static_assert(checkFoldResult<NotBadFoldResult>(
    [](size_t n) constexpr {
      return __builtin_memcmp(&ui0000FEFF, &usFEFF, n);
    },
    withN<1u>(), withExpected<LittleEndian ? 0 : -1>()));

constexpr unsigned int ui08038700 = 0x08038700u;
constexpr unsigned int ui08048600 = 0x08048600u;
static_assert(checkFoldResult<NotBadFoldResult>(
    [](size_t n) constexpr {
      return __builtin_memcmp(&ui08038700, &ui08048600, n);
    },
    withN<sizeof(int)>(), withExpected<LittleEndian ? 1 : -1>()));
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

  constexpr const char8_t *kU8Str = u8"abca\xff\0d";
  constexpr char8_t kU8Foo[] = {u8'f', u8'o', u8'o'};
  static_assert(__builtin_memchr(kU8Str, u8'a', 0) == nullptr);
  static_assert(__builtin_memchr(kU8Str, u8'a', 1) == kU8Str);
  static_assert(__builtin_memchr(kU8Str, u8'\0', 5) == nullptr);
  static_assert(__builtin_memchr(kU8Str, u8'\0', 6) == kU8Str + 5);
  static_assert(__builtin_memchr(kU8Str, u8'\xff', 8) == kU8Str + 4);
  static_assert(__builtin_memchr(kU8Str, u8'\xff' + 256, 8) == kU8Str + 4);
  static_assert(__builtin_memchr(kU8Str, u8'\xff' - 256, 8) == kU8Str + 4);
  static_assert(__builtin_memchr(kU8Foo, u8'x', 3) == nullptr);
  static_assert(__builtin_memchr(kU8Foo, u8'x', 4) == nullptr); // expected-error {{not an integral constant}} expected-note {{dereferenced one-past-the-end}}
  static_assert(__builtin_memchr(nullptr, u8'x', 3) == nullptr); // expected-error {{not an integral constant}} expected-note {{dereferenced null}}
  static_assert(__builtin_memchr(nullptr, u8'x', 0) == nullptr); // FIXME: Should we reject this?

  extern struct Incomplete incomplete;
  static_assert(__builtin_memchr(&incomplete, 0, 0u) == nullptr);
  static_assert(__builtin_memchr(&incomplete, 0, 1u) == nullptr); // expected-error {{not an integral constant}} expected-note {{read of incomplete type 'struct Incomplete'}}

  const unsigned char &u1 = 0xf0;
  auto &&i1 = (const signed char []){-128}; // expected-warning {{compound literals are a C99-specific feature}}
  static_assert(__builtin_memchr(&u1, -(0x0f + 1), 1) == &u1);
  static_assert(__builtin_memchr(i1, 0x80, 1) == i1);

  enum class E : unsigned char {};
  struct EPair { E e, f; };
  constexpr EPair ee{E{240}};
  static_assert(__builtin_memchr(&ee.e, 240, 1) == &ee.e); // expected-error {{constant}} expected-note {{not supported}}

  constexpr bool kBool[] = {false, true, false};
  constexpr const bool *const kBoolPastTheEndPtr = kBool + 3;
  static_assert(sizeof(bool) != 1u || __builtin_memchr(kBoolPastTheEndPtr - 3, 1, 99) == kBool + 1); // expected-error {{constant}} expected-note {{not supported}}
  static_assert(sizeof(bool) != 1u || __builtin_memchr(kBool + 1, 0, 99) == kBoolPastTheEndPtr - 1); // expected-error {{constant}} expected-note {{not supported}}
  static_assert(sizeof(bool) != 1u || __builtin_memchr(kBoolPastTheEndPtr - 3, -1, 3) == nullptr); // expected-error {{constant}} expected-note {{not supported}}
  static_assert(sizeof(bool) != 1u || __builtin_memchr(kBoolPastTheEndPtr, 0, 1) == nullptr); // expected-error {{constant}} expected-note {{not supported}}

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

namespace MultibyteElementTests::Memchr {
constexpr unsigned int u04030201 = 0x04030201;
static_assert(checkFoldResult<NotBadFoldResult>(
    [](size_t n) constexpr {
      return __builtin_memchr(&u04030201, *ByteOrderString - '0', n);
    },
    withN<1u>(), withExpected<&u04030201>()));

constexpr unsigned int uED = 0xEDU;
static_assert(checkFoldResult<NotBadFoldResult>(
    [](size_t n) constexpr {
      return __builtin_memchr(&uED, 0xED, n);
    },
    withN<1u>(), withExpected<LittleEndian ? &uED : nullptr>()));
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
    // expected-note@-2 {{source is not a contiguous array of at least 2 elements of type 'wchar_t'}}
    // expected-note@-3 {{destination is not a contiguous array of at least 3 elements of type 'wchar_t'}}
    return result(arr);
  }
  constexpr int test_wmemmove(int a, int b, int n) {
    wchar_t arr[4] = {1, 2, 3, 4};
    __builtin_wmemmove(arr + a, arr + b, n);
    // expected-note@-1 {{source is not a contiguous array of at least 2 elements of type 'wchar_t'}}
    // expected-note@-2 {{destination is not a contiguous array of at least 3 elements of type 'wchar_t'}}
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

#define fold(x) (__builtin_constant_p(0) ? (x) : (x))

  wchar_t global;
  constexpr wchar_t *null = 0;
  static_assert(__builtin_memcpy(&global, null, sizeof(wchar_t))); // expected-error {{}} expected-note {{source of 'memcpy' is nullptr}}
  static_assert(__builtin_memmove(&global, null, sizeof(wchar_t))); // expected-error {{}} expected-note {{source of 'memmove' is nullptr}}
  static_assert(__builtin_wmemcpy(&global, null, sizeof(wchar_t))); // expected-error {{}} expected-note {{source of 'wmemcpy' is nullptr}}
  static_assert(__builtin_wmemmove(&global, null, sizeof(wchar_t))); // expected-error {{}} expected-note {{source of 'wmemmove' is nullptr}}
  static_assert(__builtin_memcpy(null, &global, sizeof(wchar_t))); // expected-error {{}} expected-note {{destination of 'memcpy' is nullptr}}
  static_assert(__builtin_memmove(null, &global, sizeof(wchar_t))); // expected-error {{}} expected-note {{destination of 'memmove' is nullptr}}
  static_assert(__builtin_wmemcpy(null, &global, sizeof(wchar_t))); // expected-error {{}} expected-note {{destination of 'wmemcpy' is nullptr}}
  static_assert(__builtin_wmemmove(null, &global, sizeof(wchar_t))); // expected-error {{}} expected-note {{destination of 'wmemmove' is nullptr}}
  static_assert(__builtin_memcpy(&global, fold((wchar_t*)123), sizeof(wchar_t))); // expected-error {{}} expected-note {{source of 'memcpy' is (void *)123}}
  static_assert(__builtin_memcpy(fold(reinterpret_cast<wchar_t*>(123)), &global, sizeof(wchar_t))); // expected-error {{}} expected-note {{destination of 'memcpy' is (void *)123}}
  constexpr struct Incomplete *null_incomplete = 0;
  static_assert(__builtin_memcpy(null_incomplete, null_incomplete, sizeof(wchar_t))); // expected-error {{}} expected-note {{source of 'memcpy' is nullptr}}

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

  // Check that when address-of an array is passed to a tested function the
  // array can be fully copied.
  constexpr int test_address_of_const_array_type() {
    int arr[4] = {1, 2, 3, 4};
    __builtin_memmove(&arr, &arr, sizeof(arr));
    return arr[0] * 1000 + arr[1] * 100 + arr[2] * 10 + arr[3];
  }
  static_assert(test_address_of_const_array_type() == 1234);

  // Check that an incomplete array is rejected.
  constexpr int test_incomplete_array_type() { // expected-error {{never produces a constant}}
    extern int arr[];
    __builtin_memmove(arr, arr, 4 * sizeof(arr[0]));
    // expected-note@-1 2{{'memmove' not supported: source is not a contiguous array of at least 4 elements of type 'int'}}
    return arr[0] * 1000 + arr[1] * 100 + arr[2] * 10 + arr[3];
  }
  static_assert(test_incomplete_array_type() == 1234); // expected-error {{constant}} expected-note {{in call}}

  // Check that a pointer to an incomplete array is rejected.
  constexpr int test_address_of_incomplete_array_type() { // expected-error {{never produces a constant}}
    extern int arr[];
    __builtin_memmove(&arr, &arr, 4 * sizeof(arr[0]));
    // expected-note@-1 2{{cannot constant evaluate 'memmove' between objects of incomplete type 'int[]'}}
    return arr[0] * 1000 + arr[1] * 100 + arr[2] * 10 + arr[3];
  }
  static_assert(test_address_of_incomplete_array_type() == 1234); // expected-error {{constant}} expected-note {{in call}}

  // Check that a pointer to an incomplete struct is rejected.
  constexpr bool test_address_of_incomplete_struct_type() { // expected-error {{never produces a constant}}
    struct Incomplete;
    extern Incomplete x, y;
    __builtin_memcpy(&x, &x, 4);
    // expected-note@-1 2{{cannot constant evaluate 'memcpy' between objects of incomplete type 'Incomplete'}}
    return true;
  }
  static_assert(test_address_of_incomplete_struct_type()); // expected-error {{constant}} expected-note {{in call}}
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconstant-conversion"
namespace GH64876 {
void f() {
  __builtin_strncmp(0, 0, 0xffffffffffffffff);
  __builtin_memcmp(0, 0, 0xffffffffffffffff);
  __builtin_bcmp(0, 0, 0xffffffffffffffff);
  __builtin_wmemcmp(0, 0, 0xffffffffffffffff);
  __builtin_memchr((const void*)0, 1, 0xffffffffffffffff);
  __builtin_wmemchr((const wchar_t*)0, 1, 0xffffffffffffffff);

  __builtin_strncmp(0, 0, -511LL);
  __builtin_memcmp(0, 0, -511LL);
  __builtin_bcmp(0, 0, -511LL);
  __builtin_wmemcmp(0, 0, -511LL);
  __builtin_memchr((const void*)0, 1, -511LL);
  __builtin_wmemchr((const wchar_t*)0, 1, -511LL);
}
}
#pragma clang diagnostic pop
