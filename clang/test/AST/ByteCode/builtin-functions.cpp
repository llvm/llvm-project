// RUN: %clang_cc1 -Wno-string-plus-int -fexperimental-new-constant-interpreter -triple x86_64 %s -verify=expected,both
// RUN: %clang_cc1 -Wno-string-plus-int                                         -triple x86_64 %s -verify=ref,both
//
// RUN: %clang_cc1 -Wno-string-plus-int -fexperimental-new-constant-interpreter -triple i686 %s -verify=expected,both
// RUN: %clang_cc1 -Wno-string-plus-int                                         -triple i686 %s -verify=ref,both
//
// RUN: %clang_cc1 -std=c++20 -Wno-string-plus-int -fexperimental-new-constant-interpreter -triple x86_64 %s -verify=expected,both
// RUN: %clang_cc1 -std=c++20 -Wno-string-plus-int                                         -triple x86_64 %s -verify=ref,both
//
// RUN: %clang_cc1 -std=c++20 -Wno-string-plus-int -fexperimental-new-constant-interpreter -triple i686 %s -verify=expected,both
// RUN: %clang_cc1 -std=c++20 -Wno-string-plus-int                                         -triple i686 %s -verify=ref,both
//
// RUN: %clang_cc1 -triple avr -std=c++20 -Wno-string-plus-int -fexperimental-new-constant-interpreter %s -verify=expected,both
// RUN: %clang_cc1 -triple avr -std=c++20 -Wno-string-plus-int                                            -verify=ref,both %s

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define LITTLE_END 1
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define LITTLE_END 0
#else
#error "huh?"
#endif


inline constexpr void* operator new(__SIZE_TYPE__, void* p) noexcept { return p; }
namespace std {
  using size_t = decltype(sizeof(0));
  template<typename T> struct allocator {
    constexpr T *allocate(size_t N) {
      return (T*)__builtin_operator_new(sizeof(T) * N); // #alloc
    }
    constexpr void deallocate(void *p, __SIZE_TYPE__) {
      __builtin_operator_delete(p);
    }
  };
template<typename T, typename... Args>
constexpr T* construct_at(T* p, Args&&... args) { return ::new((void*)p) T(static_cast<Args&&>(args)...); }

  template<typename T>
  constexpr void destroy_at(T* p) {
    p->~T();
  }
}

extern "C" {
  typedef decltype(sizeof(int)) size_t;
  extern size_t wcslen(const wchar_t *p);
  extern void *memchr(const void *s, int c, size_t n);
  extern char *strchr(const char *s, int c);
  extern wchar_t *wmemchr(const wchar_t *s, wchar_t c, size_t n);
  extern wchar_t *wcschr(const wchar_t *s, wchar_t c);
  extern int wcscmp(const wchar_t *s1, const wchar_t *s2);
  extern int wcsncmp(const wchar_t *s1, const wchar_t *s2, size_t n);
  extern wchar_t *wmemcpy(wchar_t *d, const wchar_t *s, size_t n);
}


constexpr int test_address_of_incomplete_array_type() { // both-error {{never produces a constant expression}}
  extern int arr[];
  __builtin_memmove(&arr, &arr, 4 * sizeof(arr[0])); // both-note 2{{cannot constant evaluate 'memmove' between objects of incomplete type 'int[]'}}
  return arr[0] * 1000 + arr[1] * 100 + arr[2] * 10 + arr[3];
}
static_assert(test_address_of_incomplete_array_type() == 1234, ""); // both-error {{constant}} \
                                                                    // both-note {{in call}}


  struct NonTrivial {
    constexpr NonTrivial() : n(0) {}
    constexpr NonTrivial(const NonTrivial &) : n(1) {}
    int n;
  };
  constexpr bool test_nontrivial_memcpy() { // both-error {{never produces a constant}}
    NonTrivial arr[3] = {};
    __builtin_memcpy(arr, arr + 1, sizeof(NonTrivial)); // both-note {{non-trivially-copyable}} \
                                                        // both-note {{non-trivially-copyable}}
    return true;
  }
  static_assert(test_nontrivial_memcpy()); // both-error {{constant}} \
                                           // both-note {{in call}}

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
  static_assert(__builtin_strcmp(0, "abab") == 0, ""); // both-error {{not an integral constant}} \
                                                       // both-note {{dereferenced null}}
  static_assert(__builtin_strcmp("abab", 0) == 0, ""); // both-error {{not an integral constant}} \
                                                       // both-note {{dereferenced null}}

  static_assert(__builtin_strcmp(kFoobar, kFoobazfoobar) == -1, "");
  static_assert(__builtin_strcmp(kFoobar, kFoobazfoobar + 6) == 0, ""); // both-error {{not an integral constant}} \
                                                                        // both-note {{dereferenced one-past-the-end}}

  /// Used to assert because we're passing a dummy pointer to
  /// __builtin_strcmp() when evaluating the return statement.
  constexpr bool char_memchr_mutable() {
    char buffer[] = "mutable";
    return __builtin_strcmp(buffer, "mutable") == 0;
  }
  static_assert(char_memchr_mutable(), "");

  static_assert(__builtin_strncmp("abaa", "abba", 5) == -1);
  static_assert(__builtin_strncmp("abaa", "abba", 4) == -1);
  static_assert(__builtin_strncmp("abaa", "abba", 3) == -1);
  static_assert(__builtin_strncmp("abaa", "abba", 2) == 0);
  static_assert(__builtin_strncmp("abaa", "abba", 1) == 0);
  static_assert(__builtin_strncmp("abaa", "abba", 0) == 0);
  static_assert(__builtin_strncmp(0, 0, 0) == 0);
  static_assert(__builtin_strncmp("abab\0banana", "abab\0canada", 100) == 0);
}

namespace WcsCmp {
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
  static_assert(__builtin_wcscmp(0, L"abab") == 0); // both-error {{not an integral constant}} \
                                                    // both-note {{dereferenced null}}
  static_assert(__builtin_wcscmp(L"abab", 0) == 0); // both-error {{not an integral constant}} \
                                                    // both-note {{dereferenced null}}

  static_assert(__builtin_wcscmp(kFoobar, kFoobazfoobar) == -1);
  static_assert(__builtin_wcscmp(kFoobar, kFoobazfoobar + 6) == 0); // both-error {{not an integral constant}} \
                                                                    // both-note {{dereferenced one-past-the-end}}

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
  static_assert(__builtin_wcsncmp(kFoobar, kFoobazfoobar, 7) == -1);
  static_assert(__builtin_wcsncmp(kFoobar, kFoobazfoobar + 6, 6) == 0);
  static_assert(__builtin_wcsncmp(kFoobar, kFoobazfoobar + 6, 7) == 0); // both-error {{not an integral constant}} \
                                                                        // both-note {{dereferenced one-past-the-end}}
}

/// Copied from constant-expression-cxx11.cpp
namespace strlen {
constexpr const char *a = "foo\0quux";
  constexpr char b[] = "foo\0quux";
  constexpr int f() { return 'u'; }
  constexpr char c[] = { 'f', 'o', 'o', 0, 'q', f(), 'u', 'x', 0 };

  static_assert(__builtin_strlen("foo") == 3, "");
  static_assert(__builtin_strlen("foo\0quux") == 3, "");
  static_assert(__builtin_strlen("foo\0quux" + 4) == 4, "");

  constexpr bool check(const char *p) {
    return __builtin_strlen(p) == 3 &&
           __builtin_strlen(p + 1) == 2 &&
           __builtin_strlen(p + 2) == 1 &&
           __builtin_strlen(p + 3) == 0 &&
           __builtin_strlen(p + 4) == 4 &&
           __builtin_strlen(p + 5) == 3 &&
           __builtin_strlen(p + 6) == 2 &&
           __builtin_strlen(p + 7) == 1 &&
           __builtin_strlen(p + 8) == 0;
  }

  static_assert(check(a), "");
  static_assert(check(b), "");
  static_assert(check(c), "");

  constexpr int over1 = __builtin_strlen(a + 9); // both-error {{constant expression}} \
                                                 // both-note {{one-past-the-end}}
  constexpr int over2 = __builtin_strlen(b + 9); // both-error {{constant expression}} \
                                                 // both-note {{one-past-the-end}}
  constexpr int over3 = __builtin_strlen(c + 9); // both-error {{constant expression}} \
                                                 // both-note {{one-past-the-end}}

  constexpr int under1 = __builtin_strlen(a - 1); // both-error {{constant expression}} \
                                                  // both-note {{cannot refer to element -1}}
  constexpr int under2 = __builtin_strlen(b - 1); // both-error {{constant expression}} \
                                                  // both-note {{cannot refer to element -1}}
  constexpr int under3 = __builtin_strlen(c - 1); // both-error {{constant expression}} \
                                                  // both-note {{cannot refer to element -1}}

  constexpr char d[] = { 'f', 'o', 'o' }; // no nul terminator.
  constexpr int bad = __builtin_strlen(d); // both-error {{constant expression}} \
                                           // both-note {{one-past-the-end}}

  constexpr int wn = __builtin_wcslen(L"hello");
  static_assert(wn == 5);
  constexpr int wm = wcslen(L"hello"); // both-error {{constant expression}} \
                                       // both-note {{non-constexpr function 'wcslen' cannot be used in a constant expression}}

  int arr[3]; // both-note {{here}}
  int wk = arr[wcslen(L"hello")]; // both-warning {{array index 5}}
}

namespace nan {
  constexpr double NaN1 = __builtin_nan("");

  /// The current interpreter does not accept this, but it should.
  constexpr float NaN2 = __builtin_nans([](){return "0xAE98";}()); // ref-error {{must be initialized by a constant expression}}
#if __cplusplus < 201703L
  // expected-error@-2 {{must be initialized by a constant expression}}
#endif

  constexpr double NaN3 = __builtin_nan("foo"); // both-error {{must be initialized by a constant expression}}
  constexpr float NaN4 = __builtin_nanf("");
  constexpr long double NaN5 = __builtin_nanf128("");

  /// FIXME: This should be accepted by the current interpreter as well.
  constexpr char f[] = {'0', 'x', 'A', 'E', '\0'};
  constexpr double NaN6 = __builtin_nan(f); // ref-error {{must be initialized by a constant expression}}

  /// FIXME: Current interpreter misses diagnostics.
  constexpr char f2[] = {'0', 'x', 'A', 'E'}; /// No trailing 0 byte.
  constexpr double NaN7 = __builtin_nan(f2); // both-error {{must be initialized by a constant expression}} \
                                             // expected-note {{read of dereferenced one-past-the-end pointer}}
  static_assert(!__builtin_issignaling(__builtin_nan("")), "");
  static_assert(__builtin_issignaling(__builtin_nans("")), "");
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

#ifndef __AVR__
  static_assert(__builtin_issubnormal(0x1p-1070), "");
#endif
  static_assert(!__builtin_issubnormal(__builtin_inf()), "");

  static_assert(__builtin_iszero(0.0), "");
  static_assert(!__builtin_iszero(__builtin_inf()), "");

  static_assert(__builtin_issignaling(__builtin_nans("")), "");
  static_assert(!__builtin_issignaling(__builtin_inf()), "");
}

namespace isfpclass {
  char isfpclass_inf_pos_0[__builtin_isfpclass(__builtin_inf(), 0x0200) ? 1 : -1]; // fcPosInf
  char isfpclass_inf_pos_1[!__builtin_isfpclass(__builtin_inff(), 0x0004) ? 1 : -1]; // fcNegInf
  char isfpclass_inf_pos_2[__builtin_isfpclass(__builtin_infl(), 0x0207) ? 1 : -1]; // fcSNan|fcQNan|fcNegInf|fcPosInf
  char isfpclass_inf_pos_3[!__builtin_isfpclass(__builtin_inf(), 0x01F8) ? 1 : -1]; // fcFinite
  char isfpclass_pos_0    [__builtin_isfpclass(1.0, 0x0100) ? 1 : -1]; // fcPosNormal
  char isfpclass_pos_1    [!__builtin_isfpclass(1.0f, 0x0008) ? 1 : -1]; // fcNegNormal
  char isfpclass_pos_2    [__builtin_isfpclass(1.0L, 0x01F8) ? 1 : -1]; // fcFinite
  char isfpclass_pos_3    [!__builtin_isfpclass(1.0, 0x0003) ? 1 : -1]; // fcSNan|fcQNan
#ifndef __AVR__
  char isfpclass_pdenorm_0[__builtin_isfpclass(1.0e-40f, 0x0080) ? 1 : -1]; // fcPosSubnormal
  char isfpclass_pdenorm_1[__builtin_isfpclass(1.0e-310, 0x01F8) ? 1 : -1]; // fcFinite
  char isfpclass_pdenorm_2[!__builtin_isfpclass(1.0e-40f, 0x003C) ? 1 : -1]; // fcNegative
  char isfpclass_pdenorm_3[!__builtin_isfpclass(1.0e-310, 0x0207) ? 1 : -1]; // ~fcFinite
#endif
  char isfpclass_pzero_0  [__builtin_isfpclass(0.0f, 0x0060) ? 1 : -1]; // fcZero
  char isfpclass_pzero_1  [__builtin_isfpclass(0.0, 0x01F8) ? 1 : -1]; // fcFinite
  char isfpclass_pzero_2  [!__builtin_isfpclass(0.0L, 0x0020) ? 1 : -1]; // fcNegZero
  char isfpclass_pzero_3  [!__builtin_isfpclass(0.0, 0x0003) ? 1 : -1]; // fcNan
  char isfpclass_nzero_0  [__builtin_isfpclass(-0.0f, 0x0060) ? 1 : -1]; // fcZero
  char isfpclass_nzero_1  [__builtin_isfpclass(-0.0, 0x01F8) ? 1 : -1]; // fcFinite
  char isfpclass_nzero_2  [!__builtin_isfpclass(-0.0L, 0x0040) ? 1 : -1]; // fcPosZero
  char isfpclass_nzero_3  [!__builtin_isfpclass(-0.0, 0x0003) ? 1 : -1]; // fcNan
  char isfpclass_ndenorm_0[__builtin_isfpclass(-1.0e-40f, 0x0010) ? 1 : -1]; // fcNegSubnormal
  char isfpclass_ndenorm_2[!__builtin_isfpclass(-1.0e-40f, 0x03C0) ? 1 : -1]; // fcPositive
#ifndef __AVR__
  char isfpclass_ndenorm_1[__builtin_isfpclass(-1.0e-310, 0x01F8) ? 1 : -1]; // fcFinite
  char isfpclass_ndenorm_3[!__builtin_isfpclass(-1.0e-310, 0x0207) ? 1 : -1]; // ~fcFinite
#endif
  char isfpclass_neg_0    [__builtin_isfpclass(-1.0, 0x0008) ? 1 : -1]; // fcNegNormal
  char isfpclass_neg_1    [!__builtin_isfpclass(-1.0f, 0x00100) ? 1 : -1]; // fcPosNormal
  char isfpclass_neg_2    [__builtin_isfpclass(-1.0L, 0x01F8) ? 1 : -1]; // fcFinite
  char isfpclass_neg_3    [!__builtin_isfpclass(-1.0, 0x0003) ? 1 : -1]; // fcSNan|fcQNan
  char isfpclass_inf_neg_0[__builtin_isfpclass(-__builtin_inf(), 0x0004) ? 1 : -1]; // fcNegInf
  char isfpclass_inf_neg_1[!__builtin_isfpclass(-__builtin_inff(), 0x0200) ? 1 : -1]; // fcPosInf
  char isfpclass_inf_neg_2[__builtin_isfpclass(-__builtin_infl(), 0x0207) ? 1 : -1]; // ~fcFinite
  char isfpclass_inf_neg_3[!__builtin_isfpclass(-__builtin_inf(), 0x03C0) ? 1 : -1]; // fcPositive
  char isfpclass_qnan_0   [__builtin_isfpclass(__builtin_nan(""), 0x0002) ? 1 : -1]; // fcQNan
  char isfpclass_qnan_1   [!__builtin_isfpclass(__builtin_nanf(""), 0x0001) ? 1 : -1]; // fcSNan
  char isfpclass_qnan_2   [__builtin_isfpclass(__builtin_nanl(""), 0x0207) ? 1 : -1]; // ~fcFinite
  char isfpclass_qnan_3   [!__builtin_isfpclass(__builtin_nan(""), 0x01F8) ? 1 : -1]; // fcFinite
  char isfpclass_snan_0   [__builtin_isfpclass(__builtin_nansf(""), 0x0001) ? 1 : -1]; // fcSNan
  char isfpclass_snan_1   [!__builtin_isfpclass(__builtin_nans(""), 0x0002) ? 1 : -1]; // fcQNan
  char isfpclass_snan_2   [__builtin_isfpclass(__builtin_nansl(""), 0x0207) ? 1 : -1]; // ~fcFinite
  char isfpclass_snan_3   [!__builtin_isfpclass(__builtin_nans(""), 0x01F8) ? 1 : -1]; // fcFinite
}

namespace signbit {
  static_assert(
    !__builtin_signbit(1.0) && __builtin_signbit(-1.0) && !__builtin_signbit(0.0) && __builtin_signbit(-0.0) &&
    !__builtin_signbitf(1.0f) && __builtin_signbitf(-1.0f) && !__builtin_signbitf(0.0f) && __builtin_signbitf(-0.0f) &&
    !__builtin_signbitl(1.0L) && __builtin_signbitf(-1.0L) && !__builtin_signbitf(0.0L) && __builtin_signbitf(-0.0L) &&
    !__builtin_signbit(1.0f) && __builtin_signbit(-1.0f) && !__builtin_signbit(0.0f) && __builtin_signbit(-0.0f) &&
    !__builtin_signbit(1.0L) && __builtin_signbit(-1.0L) && !__builtin_signbit(0.0L) && __builtin_signbit(-0.0L) &&
    true, ""
  );
}

namespace floating_comparison {
#define LESS(X, Y) \
  !__builtin_isgreater(X, Y) && __builtin_isgreater(Y, X) &&             \
  !__builtin_isgreaterequal(X, Y) && __builtin_isgreaterequal(Y, X) &&   \
  __builtin_isless(X, Y) && !__builtin_isless(Y, X) &&                   \
  __builtin_islessequal(X, Y) && !__builtin_islessequal(Y, X) &&         \
  __builtin_islessgreater(X, Y) && __builtin_islessgreater(Y, X) &&      \
  !__builtin_isunordered(X, Y) && !__builtin_isunordered(Y, X)
#define EQUAL(X, Y) \
  !__builtin_isgreater(X, Y) && !__builtin_isgreater(Y, X) &&            \
  __builtin_isgreaterequal(X, Y) && __builtin_isgreaterequal(Y, X) &&    \
  !__builtin_isless(X, Y) && !__builtin_isless(Y, X) &&                  \
  __builtin_islessequal(X, Y) && __builtin_islessequal(Y, X) &&          \
  !__builtin_islessgreater(X, Y) && !__builtin_islessgreater(Y, X) &&    \
  !__builtin_isunordered(X, Y) && !__builtin_isunordered(Y, X)
#define UNORDERED(X, Y) \
  !__builtin_isgreater(X, Y) && !__builtin_isgreater(Y, X) &&            \
  !__builtin_isgreaterequal(X, Y) && !__builtin_isgreaterequal(Y, X) &&  \
  !__builtin_isless(X, Y) && !__builtin_isless(Y, X) &&                  \
  !__builtin_islessequal(X, Y) && !__builtin_islessequal(Y, X) &&        \
  !__builtin_islessgreater(X, Y) && !__builtin_islessgreater(Y, X) &&    \
  __builtin_isunordered(X, Y) && __builtin_isunordered(Y, X)

  static_assert(LESS(0.0, 1.0));
  static_assert(LESS(0.0, __builtin_inf()));
  static_assert(LESS(0.0f, 1.0f));
  static_assert(LESS(0.0f, __builtin_inff()));
  static_assert(LESS(0.0L, 1.0L));
  static_assert(LESS(0.0L, __builtin_infl()));

  static_assert(EQUAL(1.0, 1.0));
  static_assert(EQUAL(0.0, -0.0));
  static_assert(EQUAL(1.0f, 1.0f));
  static_assert(EQUAL(0.0f, -0.0f));
  static_assert(EQUAL(1.0L, 1.0L));
  static_assert(EQUAL(0.0L, -0.0L));

  static_assert(UNORDERED(__builtin_nan(""), 1.0));
  static_assert(UNORDERED(__builtin_nan(""), __builtin_inf()));
  static_assert(UNORDERED(__builtin_nanf(""), 1.0f));
  static_assert(UNORDERED(__builtin_nanf(""), __builtin_inff()));
  static_assert(UNORDERED(__builtin_nanl(""), 1.0L));
  static_assert(UNORDERED(__builtin_nanl(""), __builtin_infl()));
}

namespace fpclassify {
  char classify_nan     [__builtin_fpclassify(+1, -1, -1, -1, -1, __builtin_nan(""))];
  char classify_snan    [__builtin_fpclassify(+1, -1, -1, -1, -1, __builtin_nans(""))];
  char classify_inf     [__builtin_fpclassify(-1, +1, -1, -1, -1, __builtin_inf())];
  char classify_neg_inf [__builtin_fpclassify(-1, +1, -1, -1, -1, -__builtin_inf())];
  char classify_normal  [__builtin_fpclassify(-1, -1, +1, -1, -1, 1.539)];
#ifndef __AVR__
  char classify_normal2 [__builtin_fpclassify(-1, -1, +1, -1, -1, 1e-307)];
  char classify_denorm  [__builtin_fpclassify(-1, -1, -1, +1, -1, 1e-308)];
  char classify_denorm2 [__builtin_fpclassify(-1, -1, -1, +1, -1, -1e-308)];
#endif
  char classify_zero    [__builtin_fpclassify(-1, -1, -1, -1, +1, 0.0)];
  char classify_neg_zero[__builtin_fpclassify(-1, -1, -1, -1, +1, -0.0)];
  char classify_subnorm [__builtin_fpclassify(-1, -1, -1, +1, -1, 1.0e-38f)];
}

namespace abs {
  static_assert(__builtin_abs(14) == 14, "");
  static_assert(__builtin_labs(14L) == 14L, "");
  static_assert(__builtin_llabs(14LL) == 14LL, "");
  static_assert(__builtin_abs(-14) == 14, "");
  static_assert(__builtin_labs(-0x14L) == 0x14L, "");
  static_assert(__builtin_llabs(-0x141414141414LL) == 0x141414141414LL, "");
#define BITSIZE(x) (sizeof(x) * 8)
  constexpr int abs4 = __builtin_abs(1 << (BITSIZE(int) - 1)); // both-error {{must be initialized by a constant expression}}
  constexpr long abs6 = __builtin_labs(1L << (BITSIZE(long) - 1)); // both-error {{must be initialized by a constant expression}}
  constexpr long long abs8 = __builtin_llabs(1LL << (BITSIZE(long long) - 1)); // both-error {{must be initialized by a constant expression}}
#undef BITSIZE
} // namespace abs

namespace fabs {
  static_assert(__builtin_fabs(-14.0) == 14.0, "");
}

namespace std {
struct source_location {
  struct __impl {
    unsigned int _M_line;
    const char *_M_file_name;
    signed char _M_column;
    const char *_M_function_name;
  };
  using BuiltinT = decltype(__builtin_source_location()); // OK.
};
}

namespace SourceLocation {
  constexpr auto A = __builtin_source_location();
  static_assert(A->_M_line == __LINE__ -1, "");
  static_assert(A->_M_column == 22, "");
  static_assert(__builtin_strcmp(A->_M_function_name, "") == 0, "");
  static_assert(__builtin_strcmp(A->_M_file_name, __FILE__) == 0, "");

  static_assert(__builtin_LINE() == __LINE__, "");

  struct Foo {
    int a = __builtin_LINE();
  };

  static_assert(Foo{}.a == __LINE__, "");

  struct AA {
    int n = __builtin_LINE();
  };
  struct B {
    AA a = {};
  };
  constexpr void f() {
    constexpr B c = {};
    static_assert(c.a.n == __LINE__ - 1, "");
  }
}

#define BITSIZE(x) (sizeof(x) * 8)
namespace popcount {
  static_assert(__builtin_popcount(~0u) == __CHAR_BIT__ * sizeof(unsigned int), "");
  static_assert(__builtin_popcount(0) == 0, "");
  static_assert(__builtin_popcountl(~0ul) == __CHAR_BIT__ * sizeof(unsigned long), "");
  static_assert(__builtin_popcountl(0) == 0, "");
  static_assert(__builtin_popcountll(~0ull) == __CHAR_BIT__ * sizeof(unsigned long long), "");
  static_assert(__builtin_popcountll(0) == 0, "");
  static_assert(__builtin_popcountg((unsigned char)~0) == __CHAR_BIT__ * sizeof(unsigned char), "");
  static_assert(__builtin_popcountg((unsigned char)0) == 0, "");
  static_assert(__builtin_popcountg((unsigned short)~0) == __CHAR_BIT__ * sizeof(unsigned short), "");
  static_assert(__builtin_popcountg((unsigned short)0) == 0, "");
  static_assert(__builtin_popcountg(~0u) == __CHAR_BIT__ * sizeof(unsigned int), "");
  static_assert(__builtin_popcountg(0u) == 0, "");
  static_assert(__builtin_popcountg(~0ul) == __CHAR_BIT__ * sizeof(unsigned long), "");
  static_assert(__builtin_popcountg(0ul) == 0, "");
  static_assert(__builtin_popcountg(~0ull) == __CHAR_BIT__ * sizeof(unsigned long long), "");
  static_assert(__builtin_popcountg(0ull) == 0, "");
#ifdef __SIZEOF_INT128__
  static_assert(__builtin_popcountg(~(unsigned __int128)0) == __CHAR_BIT__ * sizeof(unsigned __int128), "");
  static_assert(__builtin_popcountg((unsigned __int128)0) == 0, "");
#endif
#ifndef __AVR__
  static_assert(__builtin_popcountg(~(unsigned _BitInt(128))0) == __CHAR_BIT__ * sizeof(unsigned _BitInt(128)), "");
  static_assert(__builtin_popcountg((unsigned _BitInt(128))0) == 0, "");
#endif

  /// From test/Sema/constant-builtins-2.c
  char popcount1[__builtin_popcount(0) == 0 ? 1 : -1];
  char popcount2[__builtin_popcount(0xF0F0) == 8 ? 1 : -1];
  char popcount3[__builtin_popcount(~0) == BITSIZE(int) ? 1 : -1];
  char popcount4[__builtin_popcount(~0L) == BITSIZE(int) ? 1 : -1];
  char popcount5[__builtin_popcountl(0L) == 0 ? 1 : -1];
  char popcount6[__builtin_popcountl(0xF0F0L) == 8 ? 1 : -1];
  char popcount7[__builtin_popcountl(~0L) == BITSIZE(long) ? 1 : -1];
  char popcount8[__builtin_popcountll(0LL) == 0 ? 1 : -1];
  char popcount9[__builtin_popcountll(0xF0F0LL) == 8 ? 1 : -1];
  char popcount10[__builtin_popcountll(~0LL) == BITSIZE(long long) ? 1 : -1];
  char popcount11[__builtin_popcountg(0U) == 0 ? 1 : -1];
  char popcount12[__builtin_popcountg(0xF0F0U) == 8 ? 1 : -1];
  char popcount13[__builtin_popcountg(~0U) == BITSIZE(int) ? 1 : -1];
  char popcount14[__builtin_popcountg(~0UL) == BITSIZE(long) ? 1 : -1];
  char popcount15[__builtin_popcountg(~0ULL) == BITSIZE(long long) ? 1 : -1];
#ifdef __SIZEOF_INT128__
  char popcount16[__builtin_popcountg(~(unsigned __int128)0) == BITSIZE(__int128) ? 1 : -1];
#endif
#ifndef __AVR__
  char popcount17[__builtin_popcountg(~(unsigned _BitInt(128))0) == BITSIZE(_BitInt(128)) ? 1 : -1];
#endif
}

namespace parity {
  /// From test/Sema/constant-builtins-2.c
  char parity1[__builtin_parity(0) == 0 ? 1 : -1];
  char parity2[__builtin_parity(0xb821) == 0 ? 1 : -1];
  char parity3[__builtin_parity(0xb822) == 0 ? 1 : -1];
  char parity4[__builtin_parity(0xb823) == 1 ? 1 : -1];
  char parity5[__builtin_parity(0xb824) == 0 ? 1 : -1];
  char parity6[__builtin_parity(0xb825) == 1 ? 1 : -1];
  char parity7[__builtin_parity(0xb826) == 1 ? 1 : -1];
  char parity8[__builtin_parity(~0) == 0 ? 1 : -1];
  char parity9[__builtin_parityl(1L << (BITSIZE(long) - 1)) == 1 ? 1 : -1];
  char parity10[__builtin_parityll(1LL << (BITSIZE(long long) - 1)) == 1 ? 1 : -1];
}

namespace clrsb {
  char clrsb1[__builtin_clrsb(0) == BITSIZE(int) - 1 ? 1 : -1];
  char clrsb2[__builtin_clrsbl(0L) == BITSIZE(long) - 1 ? 1 : -1];
  char clrsb3[__builtin_clrsbll(0LL) == BITSIZE(long long) - 1 ? 1 : -1];
  char clrsb4[__builtin_clrsb(~0) == BITSIZE(int) - 1 ? 1 : -1];
  char clrsb5[__builtin_clrsbl(~0L) == BITSIZE(long) - 1 ? 1 : -1];
  char clrsb6[__builtin_clrsbll(~0LL) == BITSIZE(long long) - 1 ? 1 : -1];
  char clrsb7[__builtin_clrsb(1) == BITSIZE(int) - 2 ? 1 : -1];
  char clrsb8[__builtin_clrsb(~1) == BITSIZE(int) - 2 ? 1 : -1];
  char clrsb9[__builtin_clrsb(1 << (BITSIZE(int) - 1)) == 0 ? 1 : -1];
  char clrsb10[__builtin_clrsb(~(1 << (BITSIZE(int) - 1))) == 0 ? 1 : -1];
  char clrsb11[__builtin_clrsb(0xf) == BITSIZE(int) - 5 ? 1 : -1];
  char clrsb12[__builtin_clrsb(~0x1f) == BITSIZE(int) - 6 ? 1 : -1];
}

namespace bitreverse {
  char bitreverse1[__builtin_bitreverse8(0x01) == 0x80 ? 1 : -1];
  char bitreverse2[__builtin_bitreverse16(0x3C48) == 0x123C ? 1 : -1];
  char bitreverse3[__builtin_bitreverse32(0x12345678) == 0x1E6A2C48 ? 1 : -1];
  char bitreverse4[__builtin_bitreverse64(0x0123456789ABCDEFULL) == 0xF7B3D591E6A2C480 ? 1 : -1];
}

namespace expect {
  constexpr int a() {
    return 12;
  }
  static_assert(__builtin_expect(a(),1) == 12, "");
  static_assert(__builtin_expect_with_probability(a(), 1, 1.0) == 12, "");
}

namespace rotateleft {
  char rotateleft1[__builtin_rotateleft8(0x01, 5) == 0x20 ? 1 : -1];
  char rotateleft2[__builtin_rotateleft16(0x3210, 11) == 0x8190 ? 1 : -1];
  char rotateleft3[__builtin_rotateleft32(0x76543210, 22) == 0x841D950C ? 1 : -1];
  char rotateleft4[__builtin_rotateleft64(0xFEDCBA9876543210ULL, 55) == 0x87F6E5D4C3B2A19ULL ? 1 : -1];
}

namespace rotateright {
  char rotateright1[__builtin_rotateright8(0x01, 5) == 0x08 ? 1 : -1];
  char rotateright2[__builtin_rotateright16(0x3210, 11) == 0x4206 ? 1 : -1];
  char rotateright3[__builtin_rotateright32(0x76543210, 22) == 0x50C841D9 ? 1 : -1];
  char rotateright4[__builtin_rotateright64(0xFEDCBA9876543210ULL, 55) == 0xB97530ECA86421FDULL ? 1 : -1];
}

namespace ffs {
  char ffs1[__builtin_ffs(0) == 0 ? 1 : -1];
  char ffs2[__builtin_ffs(1) == 1 ? 1 : -1];
  char ffs3[__builtin_ffs(0xfbe71) == 1 ? 1 : -1];
  char ffs4[__builtin_ffs(0xfbe70) == 5 ? 1 : -1];
  char ffs5[__builtin_ffs(1U << (BITSIZE(int) - 1)) == BITSIZE(int) ? 1 : -1];
  char ffs6[__builtin_ffsl(0x10L) == 5 ? 1 : -1];
  char ffs7[__builtin_ffsll(0x100LL) == 9 ? 1 : -1];
}

namespace EhReturnDataRegno {
  void test11(int X) {
    switch (X) {
      case __builtin_eh_return_data_regno(0):  // constant foldable.
      break;
    }
    __builtin_eh_return_data_regno(X);  // both-error {{argument to '__builtin_eh_return_data_regno' must be a constant integer}}
  }
}

/// From test/SemaCXX/builtins.cpp
namespace test_launder {
#define TEST_TYPE(Ptr, Type) \
  static_assert(__is_same(decltype(__builtin_launder(Ptr)), Type), "expected same type")

struct Dummy {};

using FnType = int(char);
using MemFnType = int (Dummy::*)(char);
using ConstMemFnType = int (Dummy::*)() const;

void foo() {}

void test_builtin_launder_diags(void *vp, const void *cvp, FnType *fnp,
                                MemFnType mfp, ConstMemFnType cmfp, int (&Arr)[5]) {
  __builtin_launder(vp);   // both-error {{void pointer argument to '__builtin_launder' is not allowed}}
  __builtin_launder(cvp);  // both-error {{void pointer argument to '__builtin_launder' is not allowed}}
  __builtin_launder(fnp);  // both-error {{function pointer argument to '__builtin_launder' is not allowed}}
  __builtin_launder(mfp);  // both-error {{non-pointer argument to '__builtin_launder' is not allowed}}
  __builtin_launder(cmfp); // both-error {{non-pointer argument to '__builtin_launder' is not allowed}}
  (void)__builtin_launder(&fnp);
  __builtin_launder(42);      // both-error {{non-pointer argument to '__builtin_launder' is not allowed}}
  __builtin_launder(nullptr); // both-error {{non-pointer argument to '__builtin_launder' is not allowed}}
  __builtin_launder(foo);     // both-error {{function pointer argument to '__builtin_launder' is not allowed}}
  (void)__builtin_launder(Arr);
}

void test_builtin_launder(char *p, const volatile int *ip, const float *&fp,
                          double *__restrict dp) {
  int x;
  __builtin_launder(x); // both-error {{non-pointer argument to '__builtin_launder' is not allowed}}

  TEST_TYPE(p, char*);
  TEST_TYPE(ip, const volatile int*);
  TEST_TYPE(fp, const float*);
  TEST_TYPE(dp, double *__restrict);

  char *d = __builtin_launder(p);
  const volatile int *id = __builtin_launder(ip);
  int *id2 = __builtin_launder(ip); // both-error {{cannot initialize a variable of type 'int *' with an rvalue of type 'const volatile int *'}}
  const float* fd = __builtin_launder(fp);
}

void test_launder_return_type(const int (&ArrayRef)[101], int (&MArrRef)[42][13],
                              void (**&FuncPtrRef)()) {
  TEST_TYPE(ArrayRef, const int *);
  TEST_TYPE(MArrRef, int(*)[13]);
  TEST_TYPE(FuncPtrRef, void (**)());
}

template <class Tp>
constexpr Tp *test_constexpr_launder(Tp *tp) {
  return __builtin_launder(tp);
}
constexpr int const_int = 42;
constexpr int const_int2 = 101;
constexpr const int *const_ptr = test_constexpr_launder(&const_int);
static_assert(&const_int == const_ptr, "");
static_assert(const_ptr != test_constexpr_launder(&const_int2), "");

void test_non_constexpr() {
  constexpr int i = 42;                            // both-note {{address of non-static constexpr variable 'i' may differ on each invocation}}
  constexpr const int *ip = __builtin_launder(&i); // both-error {{constexpr variable 'ip' must be initialized by a constant expression}}
  // both-note@-1 {{pointer to 'i' is not a constant expression}}
}

constexpr bool test_in_constexpr(const int &i) {
  return (__builtin_launder(&i) == &i);
}

static_assert(test_in_constexpr(const_int), "");
void f() {
  constexpr int i = 42;
  static_assert(test_in_constexpr(i), "");
}

struct Incomplete; // both-note {{forward declaration}}
struct IncompleteMember {
  Incomplete &i;
};
void test_incomplete(Incomplete *i, IncompleteMember *im) {
  // both-error@+1 {{incomplete type 'Incomplete' where a complete type is required}}
  __builtin_launder(i);
  __builtin_launder(&i); // OK
  __builtin_launder(im); // OK
}

void test_noexcept(int *i) {
  static_assert(noexcept(__builtin_launder(i)), "");
}
#undef TEST_TYPE
} // end namespace test_launder


namespace clz {
  char clz1[__builtin_clz(1) == BITSIZE(int) - 1 ? 1 : -1];
  char clz2[__builtin_clz(7) == BITSIZE(int) - 3 ? 1 : -1];
  char clz3[__builtin_clz(1 << (BITSIZE(int) - 1)) == 0 ? 1 : -1];
  int clz4 = __builtin_clz(0);
  char clz5[__builtin_clzl(0xFL) == BITSIZE(long) - 4 ? 1 : -1];
  char clz6[__builtin_clzll(0xFFLL) == BITSIZE(long long) - 8 ? 1 : -1];
  char clz7[__builtin_clzs(0x1) == BITSIZE(short) - 1 ? 1 : -1];
  char clz8[__builtin_clzs(0xf) == BITSIZE(short) - 4 ? 1 : -1];
  char clz9[__builtin_clzs(0xfff) == BITSIZE(short) - 12 ? 1 : -1];

  int clz10 = __builtin_clzg((unsigned char)0);
  char clz11[__builtin_clzg((unsigned char)0, 42) == 42 ? 1 : -1];
  char clz12[__builtin_clzg((unsigned char)0x1) == BITSIZE(char) - 1 ? 1 : -1];
  char clz13[__builtin_clzg((unsigned char)0x1, 42) == BITSIZE(char) - 1 ? 1 : -1];
  char clz14[__builtin_clzg((unsigned char)0xf) == BITSIZE(char) - 4 ? 1 : -1];
  char clz15[__builtin_clzg((unsigned char)0xf, 42) == BITSIZE(char) - 4 ? 1 : -1];
  char clz16[__builtin_clzg((unsigned char)(1 << (BITSIZE(char) - 1))) == 0 ? 1 : -1];
  char clz17[__builtin_clzg((unsigned char)(1 << (BITSIZE(char) - 1)), 42) == 0 ? 1 : -1];
  int clz18 = __builtin_clzg((unsigned short)0);
  char clz19[__builtin_clzg((unsigned short)0, 42) == 42 ? 1 : -1];
  char clz20[__builtin_clzg((unsigned short)0x1) == BITSIZE(short) - 1 ? 1 : -1];
  char clz21[__builtin_clzg((unsigned short)0x1, 42) == BITSIZE(short) - 1 ? 1 : -1];
  char clz22[__builtin_clzg((unsigned short)0xf) == BITSIZE(short) - 4 ? 1 : -1];
  char clz23[__builtin_clzg((unsigned short)0xf, 42) == BITSIZE(short) - 4 ? 1 : -1];
  char clz24[__builtin_clzg((unsigned short)(1 << (BITSIZE(short) - 1))) == 0 ? 1 : -1];
  char clz25[__builtin_clzg((unsigned short)(1 << (BITSIZE(short) - 1)), 42) == 0 ? 1 : -1];
  int clz26 = __builtin_clzg(0U);
  char clz27[__builtin_clzg(0U, 42) == 42 ? 1 : -1];
  char clz28[__builtin_clzg(0x1U) == BITSIZE(int) - 1 ? 1 : -1];
  char clz29[__builtin_clzg(0x1U, 42) == BITSIZE(int) - 1 ? 1 : -1];
  char clz30[__builtin_clzg(0xfU) == BITSIZE(int) - 4 ? 1 : -1];
  char clz31[__builtin_clzg(0xfU, 42) == BITSIZE(int) - 4 ? 1 : -1];
  char clz32[__builtin_clzg(1U << (BITSIZE(int) - 1)) == 0 ? 1 : -1];
  char clz33[__builtin_clzg(1U << (BITSIZE(int) - 1), 42) == 0 ? 1 : -1];
  int clz34 = __builtin_clzg(0UL);
  char clz35[__builtin_clzg(0UL, 42) == 42 ? 1 : -1];
  char clz36[__builtin_clzg(0x1UL) == BITSIZE(long) - 1 ? 1 : -1];
  char clz37[__builtin_clzg(0x1UL, 42) == BITSIZE(long) - 1 ? 1 : -1];
  char clz38[__builtin_clzg(0xfUL) == BITSIZE(long) - 4 ? 1 : -1];
  char clz39[__builtin_clzg(0xfUL, 42) == BITSIZE(long) - 4 ? 1 : -1];
  char clz40[__builtin_clzg(1UL << (BITSIZE(long) - 1)) == 0 ? 1 : -1];
  char clz41[__builtin_clzg(1UL << (BITSIZE(long) - 1), 42) == 0 ? 1 : -1];
  int clz42 = __builtin_clzg(0ULL);
  char clz43[__builtin_clzg(0ULL, 42) == 42 ? 1 : -1];
  char clz44[__builtin_clzg(0x1ULL) == BITSIZE(long long) - 1 ? 1 : -1];
  char clz45[__builtin_clzg(0x1ULL, 42) == BITSIZE(long long) - 1 ? 1 : -1];
  char clz46[__builtin_clzg(0xfULL) == BITSIZE(long long) - 4 ? 1 : -1];
  char clz47[__builtin_clzg(0xfULL, 42) == BITSIZE(long long) - 4 ? 1 : -1];
  char clz48[__builtin_clzg(1ULL << (BITSIZE(long long) - 1)) == 0 ? 1 : -1];
  char clz49[__builtin_clzg(1ULL << (BITSIZE(long long) - 1), 42) == 0 ? 1 : -1];
#ifdef __SIZEOF_INT128__
  int clz50 = __builtin_clzg((unsigned __int128)0);
  char clz51[__builtin_clzg((unsigned __int128)0, 42) == 42 ? 1 : -1];
  char clz52[__builtin_clzg((unsigned __int128)0x1) == BITSIZE(__int128) - 1 ? 1 : -1];
  char clz53[__builtin_clzg((unsigned __int128)0x1, 42) == BITSIZE(__int128) - 1 ? 1 : -1];
  char clz54[__builtin_clzg((unsigned __int128)0xf) == BITSIZE(__int128) - 4 ? 1 : -1];
  char clz55[__builtin_clzg((unsigned __int128)0xf, 42) == BITSIZE(__int128) - 4 ? 1 : -1];
#endif
#ifndef __AVR__
  int clz58 = __builtin_clzg((unsigned _BitInt(128))0);
  char clz59[__builtin_clzg((unsigned _BitInt(128))0, 42) == 42 ? 1 : -1];
  char clz60[__builtin_clzg((unsigned _BitInt(128))0x1) == BITSIZE(_BitInt(128)) - 1 ? 1 : -1];
  char clz61[__builtin_clzg((unsigned _BitInt(128))0x1, 42) == BITSIZE(_BitInt(128)) - 1 ? 1 : -1];
  char clz62[__builtin_clzg((unsigned _BitInt(128))0xf) == BITSIZE(_BitInt(128)) - 4 ? 1 : -1];
  char clz63[__builtin_clzg((unsigned _BitInt(128))0xf, 42) == BITSIZE(_BitInt(128)) - 4 ? 1 : -1];
#endif
}

namespace ctz {
  char ctz1[__builtin_ctz(1) == 0 ? 1 : -1];
  char ctz2[__builtin_ctz(8) == 3 ? 1 : -1];
  char ctz3[__builtin_ctz(1 << (BITSIZE(int) - 1)) == BITSIZE(int) - 1 ? 1 : -1];
  int ctz4 = __builtin_ctz(0);
  char ctz5[__builtin_ctzl(0x10L) == 4 ? 1 : -1];
  char ctz6[__builtin_ctzll(0x100LL) == 8 ? 1 : -1];
  char ctz7[__builtin_ctzs(1 << (BITSIZE(short) - 1)) == BITSIZE(short) - 1 ? 1 : -1];
  int ctz8 = __builtin_ctzg((unsigned char)0);
  char ctz9[__builtin_ctzg((unsigned char)0, 42) == 42 ? 1 : -1];
  char ctz10[__builtin_ctzg((unsigned char)0x1) == 0 ? 1 : -1];
  char ctz11[__builtin_ctzg((unsigned char)0x1, 42) == 0 ? 1 : -1];
  char ctz12[__builtin_ctzg((unsigned char)0x10) == 4 ? 1 : -1];
  char ctz13[__builtin_ctzg((unsigned char)0x10, 42) == 4 ? 1 : -1];
  char ctz14[__builtin_ctzg((unsigned char)(1 << (BITSIZE(char) - 1))) == BITSIZE(char) - 1 ? 1 : -1];
  char ctz15[__builtin_ctzg((unsigned char)(1 << (BITSIZE(char) - 1)), 42) == BITSIZE(char) - 1 ? 1 : -1];
  int ctz16 = __builtin_ctzg((unsigned short)0);
  char ctz17[__builtin_ctzg((unsigned short)0, 42) == 42 ? 1 : -1];
  char ctz18[__builtin_ctzg((unsigned short)0x1) == 0 ? 1 : -1];
  char ctz19[__builtin_ctzg((unsigned short)0x1, 42) == 0 ? 1 : -1];
  char ctz20[__builtin_ctzg((unsigned short)0x10) == 4 ? 1 : -1];
  char ctz21[__builtin_ctzg((unsigned short)0x10, 42) == 4 ? 1 : -1];
  char ctz22[__builtin_ctzg((unsigned short)(1 << (BITSIZE(short) - 1))) == BITSIZE(short) - 1 ? 1 : -1];
  char ctz23[__builtin_ctzg((unsigned short)(1 << (BITSIZE(short) - 1)), 42) == BITSIZE(short) - 1 ? 1 : -1];
  int ctz24 = __builtin_ctzg(0U);
  char ctz25[__builtin_ctzg(0U, 42) == 42 ? 1 : -1];
  char ctz26[__builtin_ctzg(0x1U) == 0 ? 1 : -1];
  char ctz27[__builtin_ctzg(0x1U, 42) == 0 ? 1 : -1];
  char ctz28[__builtin_ctzg(0x10U) == 4 ? 1 : -1];
  char ctz29[__builtin_ctzg(0x10U, 42) == 4 ? 1 : -1];
  char ctz30[__builtin_ctzg(1U << (BITSIZE(int) - 1)) == BITSIZE(int) - 1 ? 1 : -1];
  char ctz31[__builtin_ctzg(1U << (BITSIZE(int) - 1), 42) == BITSIZE(int) - 1 ? 1 : -1];
  int ctz32 = __builtin_ctzg(0UL);
  char ctz33[__builtin_ctzg(0UL, 42) == 42 ? 1 : -1];
  char ctz34[__builtin_ctzg(0x1UL) == 0 ? 1 : -1];
  char ctz35[__builtin_ctzg(0x1UL, 42) == 0 ? 1 : -1];
  char ctz36[__builtin_ctzg(0x10UL) == 4 ? 1 : -1];
  char ctz37[__builtin_ctzg(0x10UL, 42) == 4 ? 1 : -1];
  char ctz38[__builtin_ctzg(1UL << (BITSIZE(long) - 1)) == BITSIZE(long) - 1 ? 1 : -1];
  char ctz39[__builtin_ctzg(1UL << (BITSIZE(long) - 1), 42) == BITSIZE(long) - 1 ? 1 : -1];
  int ctz40 = __builtin_ctzg(0ULL);
  char ctz41[__builtin_ctzg(0ULL, 42) == 42 ? 1 : -1];
  char ctz42[__builtin_ctzg(0x1ULL) == 0 ? 1 : -1];
  char ctz43[__builtin_ctzg(0x1ULL, 42) == 0 ? 1 : -1];
  char ctz44[__builtin_ctzg(0x10ULL) == 4 ? 1 : -1];
  char ctz45[__builtin_ctzg(0x10ULL, 42) == 4 ? 1 : -1];
  char ctz46[__builtin_ctzg(1ULL << (BITSIZE(long long) - 1)) == BITSIZE(long long) - 1 ? 1 : -1];
  char ctz47[__builtin_ctzg(1ULL << (BITSIZE(long long) - 1), 42) == BITSIZE(long long) - 1 ? 1 : -1];
#ifdef __SIZEOF_INT128__
  int ctz48 = __builtin_ctzg((unsigned __int128)0);
  char ctz49[__builtin_ctzg((unsigned __int128)0, 42) == 42 ? 1 : -1];
  char ctz50[__builtin_ctzg((unsigned __int128)0x1) == 0 ? 1 : -1];
  char ctz51[__builtin_ctzg((unsigned __int128)0x1, 42) == 0 ? 1 : -1];
  char ctz52[__builtin_ctzg((unsigned __int128)0x10) == 4 ? 1 : -1];
  char ctz53[__builtin_ctzg((unsigned __int128)0x10, 42) == 4 ? 1 : -1];
  char ctz54[__builtin_ctzg((unsigned __int128)1 << (BITSIZE(__int128) - 1)) == BITSIZE(__int128) - 1 ? 1 : -1];
  char ctz55[__builtin_ctzg((unsigned __int128)1 << (BITSIZE(__int128) - 1), 42) == BITSIZE(__int128) - 1 ? 1 : -1];
#endif
#ifndef __AVR__
  int ctz56 = __builtin_ctzg((unsigned _BitInt(128))0);
  char ctz57[__builtin_ctzg((unsigned _BitInt(128))0, 42) == 42 ? 1 : -1];
  char ctz58[__builtin_ctzg((unsigned _BitInt(128))0x1) == 0 ? 1 : -1];
  char ctz59[__builtin_ctzg((unsigned _BitInt(128))0x1, 42) == 0 ? 1 : -1];
  char ctz60[__builtin_ctzg((unsigned _BitInt(128))0x10) == 4 ? 1 : -1];
  char ctz61[__builtin_ctzg((unsigned _BitInt(128))0x10, 42) == 4 ? 1 : -1];
  char ctz62[__builtin_ctzg((unsigned _BitInt(128))1 << (BITSIZE(_BitInt(128)) - 1)) == BITSIZE(_BitInt(128)) - 1 ? 1 : -1];
  char ctz63[__builtin_ctzg((unsigned _BitInt(128))1 << (BITSIZE(_BitInt(128)) - 1), 42) == BITSIZE(_BitInt(128)) - 1 ? 1 : -1];
#endif
}

namespace bswap {
  extern int f(void);
  int h3 = __builtin_bswap16(0x1234) == 0x3412 ? 1 : f();
  int h4 = __builtin_bswap32(0x1234) == 0x34120000 ? 1 : f();
  int h5 = __builtin_bswap64(0x1234) == 0x3412000000000000 ? 1 : f();
}

#define CFSTR __builtin___CFStringMakeConstantString
void test7(void) {
  const void *X;
#if !defined(_AIX)
  X = CFSTR("\242"); // both-warning {{input conversion stopped}}
  X = CFSTR("\0"); // no-warning
  X = CFSTR(242); // both-error {{cannot initialize a parameter of type 'const char *' with an rvalue of type 'int'}}
  X = CFSTR("foo", "bar"); // both-error {{too many arguments to function call}}
#endif
}

/// The actual value on my machine is 22, but I have a feeling this will be different
/// on other targets, so just checking for != 0 here. Light testing is fine since
/// the actual implementation uses analyze_os_log::computeOSLogBufferLayout(), which
/// is tested elsewhere.
static_assert(__builtin_os_log_format_buffer_size("%{mask.xyz}s", "abc") != 0, "");

/// Copied from test/Sema/constant_builtins_vector.cpp.
/// Some tests are missing since we run this for multiple targets,
/// some of which do not support _BitInt.
#ifndef __AVR__
typedef _BitInt(128) BitInt128;
typedef double vector4double __attribute__((__vector_size__(32)));
typedef float vector4float __attribute__((__vector_size__(16)));
typedef long long vector4long __attribute__((__vector_size__(32)));
typedef int vector4int __attribute__((__vector_size__(16)));
typedef short vector4short __attribute__((__vector_size__(8)));
typedef char vector4char __attribute__((__vector_size__(4)));
typedef BitInt128 vector4BitInt128 __attribute__((__vector_size__(64)));
typedef double vector8double __attribute__((__vector_size__(64)));
typedef float vector8float __attribute__((__vector_size__(32)));
typedef long long vector8long __attribute__((__vector_size__(64)));
typedef int vector8int __attribute__((__vector_size__(32)));
typedef short vector8short __attribute__((__vector_size__(16)));
typedef char vector8char __attribute__((__vector_size__(8)));
typedef BitInt128 vector8BitInt128 __attribute__((__vector_size__(128)));

namespace convertvector {
  constexpr vector4double from_vector4double_to_vector4double_var =
      __builtin_convertvector((vector4double){0, 1, 2, 3}, vector4double);
  constexpr vector4float from_vector4double_to_vector4float_var =
      __builtin_convertvector((vector4double){0, 1, 2, 3}, vector4float);
  constexpr vector4long from_vector4double_to_vector4long_var =
      __builtin_convertvector((vector4double){0, 1, 2, 3}, vector4long);
  constexpr vector4int from_vector4double_to_vector4int_var =
      __builtin_convertvector((vector4double){0, 1, 2, 3}, vector4int);
  constexpr vector4short from_vector4double_to_vector4short_var =
      __builtin_convertvector((vector4double){0, 1, 2, 3}, vector4short);
  constexpr vector4char from_vector4double_to_vector4char_var =
      __builtin_convertvector((vector4double){0, 1, 2, 3}, vector4char);
  constexpr vector4BitInt128 from_vector4double_to_vector4BitInt128_var =
      __builtin_convertvector((vector4double){0, 1, 2, 3}, vector4BitInt128);
  constexpr vector4double from_vector4float_to_vector4double_var =
      __builtin_convertvector((vector4float){0, 1, 2, 3}, vector4double);
  constexpr vector4float from_vector4float_to_vector4float_var =
      __builtin_convertvector((vector4float){0, 1, 2, 3}, vector4float);
  constexpr vector4long from_vector4float_to_vector4long_var =
      __builtin_convertvector((vector4float){0, 1, 2, 3}, vector4long);
  constexpr vector4int from_vector4float_to_vector4int_var =
      __builtin_convertvector((vector4float){0, 1, 2, 3}, vector4int);
  constexpr vector4short from_vector4float_to_vector4short_var =
      __builtin_convertvector((vector4float){0, 1, 2, 3}, vector4short);
  constexpr vector4char from_vector4float_to_vector4char_var =
      __builtin_convertvector((vector4float){0, 1, 2, 3}, vector4char);
  constexpr vector4BitInt128 from_vector4float_to_vector4BitInt128_var =
      __builtin_convertvector((vector4float){0, 1, 2, 3}, vector4BitInt128);
  constexpr vector4double from_vector4long_to_vector4double_var =
      __builtin_convertvector((vector4long){0, 1, 2, 3}, vector4double);
  constexpr vector4float from_vector4long_to_vector4float_var =
      __builtin_convertvector((vector4long){0, 1, 2, 3}, vector4float);
  constexpr vector4long from_vector4long_to_vector4long_var =
      __builtin_convertvector((vector4long){0, 1, 2, 3}, vector4long);
  constexpr vector4int from_vector4long_to_vector4int_var =
      __builtin_convertvector((vector4long){0, 1, 2, 3}, vector4int);
  constexpr vector4short from_vector4long_to_vector4short_var =
      __builtin_convertvector((vector4long){0, 1, 2, 3}, vector4short);
  constexpr vector4char from_vector4long_to_vector4char_var =
      __builtin_convertvector((vector4long){0, 1, 2, 3}, vector4char);
  constexpr vector4BitInt128 from_vector4long_to_vector4BitInt128_var =
      __builtin_convertvector((vector4long){0, 1, 2, 3}, vector4BitInt128);
  constexpr vector4double from_vector4int_to_vector4double_var =
      __builtin_convertvector((vector4int){0, 1, 2, 3}, vector4double);
  constexpr vector4float from_vector4int_to_vector4float_var =
      __builtin_convertvector((vector4int){0, 1, 2, 3}, vector4float);
  constexpr vector4long from_vector4int_to_vector4long_var =
      __builtin_convertvector((vector4int){0, 1, 2, 3}, vector4long);
  constexpr vector4int from_vector4int_to_vector4int_var =
      __builtin_convertvector((vector4int){0, 1, 2, 3}, vector4int);
  constexpr vector4short from_vector4int_to_vector4short_var =
      __builtin_convertvector((vector4int){0, 1, 2, 3}, vector4short);
  constexpr vector4char from_vector4int_to_vector4char_var =
      __builtin_convertvector((vector4int){0, 1, 2, 3}, vector4char);
  constexpr vector4BitInt128 from_vector4int_to_vector4BitInt128_var =
      __builtin_convertvector((vector4int){0, 1, 2, 3}, vector4BitInt128);
  constexpr vector4double from_vector4short_to_vector4double_var =
      __builtin_convertvector((vector4short){0, 1, 2, 3}, vector4double);
  constexpr vector4float from_vector4short_to_vector4float_var =
      __builtin_convertvector((vector4short){0, 1, 2, 3}, vector4float);
  constexpr vector4long from_vector4short_to_vector4long_var =
      __builtin_convertvector((vector4short){0, 1, 2, 3}, vector4long);
  constexpr vector4int from_vector4short_to_vector4int_var =
      __builtin_convertvector((vector4short){0, 1, 2, 3}, vector4int);
  constexpr vector4short from_vector4short_to_vector4short_var =
      __builtin_convertvector((vector4short){0, 1, 2, 3}, vector4short);
  constexpr vector4char from_vector4short_to_vector4char_var =
      __builtin_convertvector((vector4short){0, 1, 2, 3}, vector4char);
  constexpr vector4BitInt128 from_vector4short_to_vector4BitInt128_var =
      __builtin_convertvector((vector4short){0, 1, 2, 3}, vector4BitInt128);
  constexpr vector4double from_vector4char_to_vector4double_var =
      __builtin_convertvector((vector4char){0, 1, 2, 3}, vector4double);
  constexpr vector4float from_vector4char_to_vector4float_var =
      __builtin_convertvector((vector4char){0, 1, 2, 3}, vector4float);
  constexpr vector4long from_vector4char_to_vector4long_var =
      __builtin_convertvector((vector4char){0, 1, 2, 3}, vector4long);
  constexpr vector4int from_vector4char_to_vector4int_var =
      __builtin_convertvector((vector4char){0, 1, 2, 3}, vector4int);
  constexpr vector4short from_vector4char_to_vector4short_var =
      __builtin_convertvector((vector4char){0, 1, 2, 3}, vector4short);
  constexpr vector4char from_vector4char_to_vector4char_var =
      __builtin_convertvector((vector4char){0, 1, 2, 3}, vector4char);
  constexpr vector8double from_vector8double_to_vector8double_var =
      __builtin_convertvector((vector8double){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8double);
  constexpr vector8float from_vector8double_to_vector8float_var =
      __builtin_convertvector((vector8double){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8float);
  constexpr vector8long from_vector8double_to_vector8long_var =
      __builtin_convertvector((vector8double){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8long);
  constexpr vector8int from_vector8double_to_vector8int_var =
      __builtin_convertvector((vector8double){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8int);
  constexpr vector8short from_vector8double_to_vector8short_var =
      __builtin_convertvector((vector8double){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8short);
  constexpr vector8char from_vector8double_to_vector8char_var =
      __builtin_convertvector((vector8double){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8char);
  constexpr vector8BitInt128 from_vector8double_to_vector8BitInt128_var =
      __builtin_convertvector((vector8double){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8BitInt128);
  constexpr vector8double from_vector8float_to_vector8double_var =
      __builtin_convertvector((vector8float){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8double);
  constexpr vector8float from_vector8float_to_vector8float_var =
      __builtin_convertvector((vector8float){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8float);
  constexpr vector8long from_vector8float_to_vector8long_var =
      __builtin_convertvector((vector8float){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8long);
  constexpr vector8int from_vector8float_to_vector8int_var =
      __builtin_convertvector((vector8float){0, 1, 2, 3, 4, 5, 6, 7}, vector8int);
  constexpr vector8short from_vector8float_to_vector8short_var =
      __builtin_convertvector((vector8float){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8short);
  constexpr vector8char from_vector8float_to_vector8char_var =
      __builtin_convertvector((vector8float){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8char);
  constexpr vector8BitInt128 from_vector8float_to_vector8BitInt128_var =
      __builtin_convertvector((vector8float){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8BitInt128);
  constexpr vector8double from_vector8long_to_vector8double_var =
      __builtin_convertvector((vector8long){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8double);
  constexpr vector8float from_vector8long_to_vector8float_var =
      __builtin_convertvector((vector8long){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8float);
  constexpr vector8long from_vector8long_to_vector8long_var =
      __builtin_convertvector((vector8long){0, 1, 2, 3, 4, 5, 6, 7}, vector8long);
  constexpr vector8int from_vector8long_to_vector8int_var =
      __builtin_convertvector((vector8long){0, 1, 2, 3, 4, 5, 6, 7}, vector8int);
  constexpr vector8short from_vector8long_to_vector8short_var =
      __builtin_convertvector((vector8long){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8short);
  constexpr vector8char from_vector8long_to_vector8char_var =
      __builtin_convertvector((vector8long){0, 1, 2, 3, 4, 5, 6, 7}, vector8char);
  constexpr vector8double from_vector8int_to_vector8double_var =
      __builtin_convertvector((vector8int){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8double);
  constexpr vector8float from_vector8int_to_vector8float_var =
      __builtin_convertvector((vector8int){0, 1, 2, 3, 4, 5, 6, 7}, vector8float);
  constexpr vector8long from_vector8int_to_vector8long_var =
      __builtin_convertvector((vector8int){0, 1, 2, 3, 4, 5, 6, 7}, vector8long);
  constexpr vector8int from_vector8int_to_vector8int_var =
      __builtin_convertvector((vector8int){0, 1, 2, 3, 4, 5, 6, 7}, vector8int);
  constexpr vector8short from_vector8int_to_vector8short_var =
      __builtin_convertvector((vector8int){0, 1, 2, 3, 4, 5, 6, 7}, vector8short);
  constexpr vector8char from_vector8int_to_vector8char_var =
      __builtin_convertvector((vector8int){0, 1, 2, 3, 4, 5, 6, 7}, vector8char);
  constexpr vector8double from_vector8short_to_vector8double_var =
      __builtin_convertvector((vector8short){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8double);
  constexpr vector8float from_vector8short_to_vector8float_var =
      __builtin_convertvector((vector8short){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8float);
  constexpr vector8long from_vector8short_to_vector8long_var =
      __builtin_convertvector((vector8short){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8long);
  constexpr vector8int from_vector8short_to_vector8int_var =
      __builtin_convertvector((vector8short){0, 1, 2, 3, 4, 5, 6, 7}, vector8int);
  constexpr vector8short from_vector8short_to_vector8short_var =
      __builtin_convertvector((vector8short){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8short);
  constexpr vector8char from_vector8short_to_vector8char_var =
      __builtin_convertvector((vector8short){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8char);

  constexpr vector8double from_vector8char_to_vector8double_var =
      __builtin_convertvector((vector8char){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8double);
  constexpr vector8float from_vector8char_to_vector8float_var =
      __builtin_convertvector((vector8char){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8float);
  constexpr vector8long from_vector8char_to_vector8long_var =
      __builtin_convertvector((vector8char){0, 1, 2, 3, 4, 5, 6, 7}, vector8long);
  constexpr vector8int from_vector8char_to_vector8int_var =
      __builtin_convertvector((vector8char){0, 1, 2, 3, 4, 5, 6, 7}, vector8int);
  constexpr vector8short from_vector8char_to_vector8short_var =
      __builtin_convertvector((vector8char){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8short);
  constexpr vector8char from_vector8char_to_vector8char_var =
      __builtin_convertvector((vector8char){0, 1, 2, 3, 4, 5, 6, 7}, vector8char);
  constexpr vector8double from_vector8BitInt128_to_vector8double_var =
      __builtin_convertvector((vector8BitInt128){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8double);
  constexpr vector8float from_vector8BitInt128_to_vector8float_var =
      __builtin_convertvector((vector8BitInt128){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8float);
  constexpr vector8long from_vector8BitInt128_to_vector8long_var =
      __builtin_convertvector((vector8BitInt128){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8long);
  constexpr vector8int from_vector8BitInt128_to_vector8int_var =
      __builtin_convertvector((vector8BitInt128){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8int);
  constexpr vector8short from_vector8BitInt128_to_vector8short_var =
      __builtin_convertvector((vector8BitInt128){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8short);
  constexpr vector8char from_vector8BitInt128_to_vector8char_var =
      __builtin_convertvector((vector8BitInt128){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8char);
  constexpr vector8BitInt128 from_vector8BitInt128_to_vector8BitInt128_var =
      __builtin_convertvector((vector8BitInt128){0, 1, 2, 3, 4, 5, 6, 7},
                              vector8BitInt128);
  static_assert(from_vector8BitInt128_to_vector8BitInt128_var[0] == 0, ""); 
  static_assert(from_vector8BitInt128_to_vector8BitInt128_var[1] == 1, ""); 
  static_assert(from_vector8BitInt128_to_vector8BitInt128_var[2] == 2, ""); 
  static_assert(from_vector8BitInt128_to_vector8BitInt128_var[3] == 3, ""); 
  static_assert(from_vector8BitInt128_to_vector8BitInt128_var[4] == 4, "");
}

namespace shufflevector {
  constexpr vector4char vector4charConst1 = {0, 1, 2, 3};
  constexpr vector4char vector4charConst2 = {4, 5, 6, 7};
  constexpr vector8char vector8intConst = {8, 9, 10, 11, 12, 13, 14, 15};
  constexpr vector4char vectorShuffle1 =
      __builtin_shufflevector(vector4charConst1, vector4charConst2, 0, 1, 2, 3);
  constexpr vector4char vectorShuffle2 =
      __builtin_shufflevector(vector4charConst1, vector4charConst2, 4, 5, 6, 7);
  constexpr vector4char vectorShuffle3 =
      __builtin_shufflevector(vector4charConst1, vector4charConst2, 0, 2, 4, 6);
  constexpr vector8char vectorShuffle4 = __builtin_shufflevector(
      vector8intConst, vector8intConst, 0, 2, 4, 6, 8, 10, 12, 14);
  constexpr vector4char vectorShuffle5 =
      __builtin_shufflevector(vector8intConst, vector8intConst, 0, 2, 4, 6);
  constexpr vector8char vectorShuffle6 = __builtin_shufflevector(
      vector4charConst1, vector4charConst2, 0, 2, 4, 6, 1, 3, 5, 7);

  static_assert(vectorShuffle6[0] == 0, "");
  static_assert(vectorShuffle6[1] == 2, "");
  static_assert(vectorShuffle6[2] == 4, "");
  static_assert(vectorShuffle6[3] == 6, "");
  static_assert(vectorShuffle6[4] == 1, "");
  static_assert(vectorShuffle6[5] == 3, "");
  static_assert(vectorShuffle6[6] == 5, "");
  static_assert(vectorShuffle6[7] == 7, "");

  constexpr vector4char  vectorShuffleFail1 = __builtin_shufflevector( // both-error {{must be initialized by a constant expression}}\
                                                                       // both-error {{index for __builtin_shufflevector not within the bounds of the input vectors; index of -1 found at position 0 is not permitted in a constexpr context}}
          vector4charConst1,
          vector4charConst2, -1, -1, -1, -1);
}

#endif

namespace FunctionStart {
  void a(void) {}
  static_assert(__builtin_function_start(a) == a, ""); // both-error {{not an integral constant expression}} \
                                                       // both-note {{comparison against opaque constant address '&__builtin_function_start(a)'}}
}

namespace BuiltinInImplicitCtor {
  constexpr struct {
    int a = __builtin_isnan(1.0);
  } Foo;
  static_assert(Foo.a == 0, "");
}

typedef double vector4double __attribute__((__vector_size__(32)));
typedef float vector4float __attribute__((__vector_size__(16)));
typedef long long vector4long __attribute__((__vector_size__(32)));
typedef int vector4int __attribute__((__vector_size__(16)));
typedef unsigned long long vector4ulong __attribute__((__vector_size__(32)));
typedef unsigned int vector4uint __attribute__((__vector_size__(16)));
typedef short vector4short __attribute__((__vector_size__(8)));
typedef char vector4char __attribute__((__vector_size__(4)));
typedef double vector8double __attribute__((__vector_size__(64)));
typedef float vector8float __attribute__((__vector_size__(32)));
typedef long long vector8long __attribute__((__vector_size__(64)));
typedef int vector8int __attribute__((__vector_size__(32)));
typedef short vector8short __attribute__((__vector_size__(16)));
typedef char vector8char __attribute__((__vector_size__(8)));

namespace RecuceAdd {
  static_assert(__builtin_reduce_add((vector4char){}) == 0);
  static_assert(__builtin_reduce_add((vector4char){1, 2, 3, 4}) == 10);
  static_assert(__builtin_reduce_add((vector4short){10, 20, 30, 40}) == 100);
  static_assert(__builtin_reduce_add((vector4int){100, 200, 300, 400}) == 1000);
  static_assert(__builtin_reduce_add((vector4long){1000, 2000, 3000, 4000}) == 10000);
  constexpr int reduceAddInt1 = __builtin_reduce_add((vector4int){~(1 << (sizeof(int) * 8 - 1)), 0, 0, 1});
  // both-error@-1 {{must be initialized by a constant expression}} \
  // both-note@-1 {{outside the range of representable values of type 'int'}}
  constexpr long long reduceAddLong1 = __builtin_reduce_add((vector4long){~(1LL << (sizeof(long long) * 8 - 1)), 0, 0, 1});
  // both-error@-1 {{must be initialized by a constant expression}} \
  // both-note@-1 {{outside the range of representable values of type 'long long'}}
  constexpr int reduceAddInt2 = __builtin_reduce_add((vector4int){(1 << (sizeof(int) * 8 - 1)), 0, 0, -1});
  // both-error@-1 {{must be initialized by a constant expression}} \
  // both-note@-1 {{outside the range of representable values of type 'int'}}
  constexpr long long reduceAddLong2 = __builtin_reduce_add((vector4long){(1LL << (sizeof(long long) * 8 - 1)), 0, 0, -1});
  // both-error@-1 {{must be initialized by a constant expression}} \
  // both-note@-1 {{outside the range of representable values of type 'long long'}}
  static_assert(__builtin_reduce_add((vector4uint){~0U, 0, 0, 1}) == 0);
  static_assert(__builtin_reduce_add((vector4ulong){~0ULL, 0, 0, 1}) == 0);


#ifdef __SIZEOF_INT128__
  typedef __int128 v4i128 __attribute__((__vector_size__(128 * 2)));
  constexpr __int128 reduceAddInt3 = __builtin_reduce_add((v4i128){});
  static_assert(reduceAddInt3 == 0);
#endif
}

namespace ReduceMul {
  static_assert(__builtin_reduce_mul((vector4char){}) == 0);
  static_assert(__builtin_reduce_mul((vector4char){1, 2, 3, 4}) == 24);
  static_assert(__builtin_reduce_mul((vector4short){1, 2, 30, 40}) == 2400);
#ifndef __AVR__
  static_assert(__builtin_reduce_mul((vector4int){10, 20, 300, 400}) == 24'000'000);
#endif
  static_assert(__builtin_reduce_mul((vector4long){1000L, 2000L, 3000L, 4000L}) == 24'000'000'000'000L);
  constexpr int reduceMulInt1 = __builtin_reduce_mul((vector4int){~(1 << (sizeof(int) * 8 - 1)), 1, 1, 2});
  // both-error@-1 {{must be initialized by a constant expression}} \
  // both-note@-1 {{outside the range of representable values of type 'int'}}
  constexpr long long reduceMulLong1 = __builtin_reduce_mul((vector4long){~(1LL << (sizeof(long long) * 8 - 1)), 1, 1, 2});
  // both-error@-1 {{must be initialized by a constant expression}} \
  // both-note@-1 {{outside the range of representable values of type 'long long'}}
  constexpr int reduceMulInt2 = __builtin_reduce_mul((vector4int){(1 << (sizeof(int) * 8 - 1)), 1, 1, 2});
  // both-error@-1 {{must be initialized by a constant expression}} \
  // both-note@-1 {{outside the range of representable values of type 'int'}}
  constexpr long long reduceMulLong2 = __builtin_reduce_mul((vector4long){(1LL << (sizeof(long long) * 8 - 1)), 1, 1, 2});
  // both-error@-1 {{must be initialized by a constant expression}} \
  // both-note@-1 {{outside the range of representable values of type 'long long'}}
  static_assert(__builtin_reduce_mul((vector4uint){~0U, 1, 1, 2}) ==
#ifdef __AVR__
      0);
#else
      (~0U - 1));
#endif
  static_assert(__builtin_reduce_mul((vector4ulong){~0ULL, 1, 1, 2}) == ~0ULL - 1);
}

namespace ReduceAnd {
  static_assert(__builtin_reduce_and((vector4char){}) == 0);
  static_assert(__builtin_reduce_and((vector4char){(char)0x11, (char)0x22, (char)0x44, (char)0x88}) == 0);
  static_assert(__builtin_reduce_and((vector4short){(short)0x1111, (short)0x2222, (short)0x4444, (short)0x8888}) == 0);
  static_assert(__builtin_reduce_and((vector4int){(int)0x11111111, (int)0x22222222, (int)0x44444444, (int)0x88888888}) == 0);
#if __INT_WIDTH__ == 32
  static_assert(__builtin_reduce_and((vector4long){(long long)0x1111111111111111L, (long long)0x2222222222222222L, (long long)0x4444444444444444L, (long long)0x8888888888888888L}) == 0L);
  static_assert(__builtin_reduce_and((vector4char){(char)-1, (char)~0x22, (char)~0x44, (char)~0x88}) == 0x11);
  static_assert(__builtin_reduce_and((vector4short){(short)~0x1111, (short)-1, (short)~0x4444, (short)~0x8888}) == 0x2222);
  static_assert(__builtin_reduce_and((vector4int){(int)~0x11111111, (int)~0x22222222, (int)-1, (int)~0x88888888}) == 0x44444444);
  static_assert(__builtin_reduce_and((vector4long){(long long)~0x1111111111111111L, (long long)~0x2222222222222222L, (long long)~0x4444444444444444L, (long long)-1}) == 0x8888888888888888L);
  static_assert(__builtin_reduce_and((vector4uint){0x11111111U, 0x22222222U, 0x44444444U, 0x88888888U}) == 0U);
  static_assert(__builtin_reduce_and((vector4ulong){0x1111111111111111UL, 0x2222222222222222UL, 0x4444444444444444UL, 0x8888888888888888UL}) == 0L);
#endif
}

namespace ReduceOr {
  static_assert(__builtin_reduce_or((vector4char){}) == 0);
  static_assert(__builtin_reduce_or((vector4char){(char)0x11, (char)0x22, (char)0x44, (char)0x88}) == (char)0xFF);
  static_assert(__builtin_reduce_or((vector4short){(short)0x1111, (short)0x2222, (short)0x4444, (short)0x8888}) == (short)0xFFFF);
  static_assert(__builtin_reduce_or((vector4int){(int)0x11111111, (int)0x22222222, (int)0x44444444, (int)0x88888888}) == (int)0xFFFFFFFF);
#if __INT_WIDTH__ == 32
  static_assert(__builtin_reduce_or((vector4long){(long long)0x1111111111111111L, (long long)0x2222222222222222L, (long long)0x4444444444444444L, (long long)0x8888888888888888L}) == (long long)0xFFFFFFFFFFFFFFFFL);
  static_assert(__builtin_reduce_or((vector4char){(char)0, (char)0x22, (char)0x44, (char)0x88}) == ~0x11);
  static_assert(__builtin_reduce_or((vector4short){(short)0x1111, (short)0, (short)0x4444, (short)0x8888}) == ~0x2222);
  static_assert(__builtin_reduce_or((vector4int){(int)0x11111111, (int)0x22222222, (int)0, (int)0x88888888}) == ~0x44444444);
  static_assert(__builtin_reduce_or((vector4long){(long long)0x1111111111111111L, (long long)0x2222222222222222L, (long long)0x4444444444444444L, (long long)0}) == ~0x8888888888888888L);
  static_assert(__builtin_reduce_or((vector4uint){0x11111111U, 0x22222222U, 0x44444444U, 0x88888888U}) == 0xFFFFFFFFU);
  static_assert(__builtin_reduce_or((vector4ulong){0x1111111111111111UL, 0x2222222222222222UL, 0x4444444444444444UL, 0x8888888888888888UL}) == 0xFFFFFFFFFFFFFFFFL);
#endif
}

namespace ReduceXor {
  static_assert(__builtin_reduce_xor((vector4char){}) == 0);
  static_assert(__builtin_reduce_xor((vector4char){(char)0x11, (char)0x22, (char)0x44, (char)0x88}) == (char)0xFF);
  static_assert(__builtin_reduce_xor((vector4short){(short)0x1111, (short)0x2222, (short)0x4444, (short)0x8888}) == (short)0xFFFF);
#if __INT_WIDTH__ == 32
  static_assert(__builtin_reduce_xor((vector4int){(int)0x11111111, (int)0x22222222, (int)0x44444444, (int)0x88888888}) == (int)0xFFFFFFFF);
  static_assert(__builtin_reduce_xor((vector4long){(long long)0x1111111111111111L, (long long)0x2222222222222222L, (long long)0x4444444444444444L, (long long)0x8888888888888888L}) == (long long)0xFFFFFFFFFFFFFFFFL);
  static_assert(__builtin_reduce_xor((vector4uint){0x11111111U, 0x22222222U, 0x44444444U, 0x88888888U}) == 0xFFFFFFFFU);
  static_assert(__builtin_reduce_xor((vector4ulong){0x1111111111111111UL, 0x2222222222222222UL, 0x4444444444444444UL, 0x8888888888888888UL}) == 0xFFFFFFFFFFFFFFFFUL);
#endif
}

namespace ElementwisePopcount {
  static_assert(__builtin_reduce_add(__builtin_elementwise_popcount((vector4int){1, 2, 3, 4})) == 5);
#if __INT_WIDTH__ == 32
  static_assert(__builtin_reduce_add(__builtin_elementwise_popcount((vector4int){0, 0xF0F0, ~0, ~0xF0F0})) == 16 * sizeof(int));
#endif
  static_assert(__builtin_reduce_add(__builtin_elementwise_popcount((vector4long){1L, 2L, 3L, 4L})) == 5L);
  static_assert(__builtin_reduce_add(__builtin_elementwise_popcount((vector4long){0L, 0xF0F0L, ~0L, ~0xF0F0L})) == 16 * sizeof(long long));
  static_assert(__builtin_reduce_add(__builtin_elementwise_popcount((vector4uint){1U, 2U, 3U, 4U})) == 5U);
  static_assert(__builtin_reduce_add(__builtin_elementwise_popcount((vector4uint){0U, 0xF0F0U, ~0U, ~0xF0F0U})) == 16 * sizeof(int));
  static_assert(__builtin_reduce_add(__builtin_elementwise_popcount((vector4ulong){1UL, 2UL, 3UL, 4UL})) == 5UL);
  static_assert(__builtin_reduce_add(__builtin_elementwise_popcount((vector4ulong){0ULL, 0xF0F0ULL, ~0ULL, ~0xF0F0ULL})) == 16 * sizeof(unsigned long long));
  static_assert(__builtin_elementwise_popcount(0) == 0);
  static_assert(__builtin_elementwise_popcount(0xF0F0) == 8);
  static_assert(__builtin_elementwise_popcount(~0) == 8 * sizeof(int));
  static_assert(__builtin_elementwise_popcount(0U) == 0);
  static_assert(__builtin_elementwise_popcount(0xF0F0U) == 8);
  static_assert(__builtin_elementwise_popcount(~0U) == 8 * sizeof(int));
  static_assert(__builtin_elementwise_popcount(0L) == 0);
  static_assert(__builtin_elementwise_popcount(0xF0F0L) == 8);
  static_assert(__builtin_elementwise_popcount(~0LL) == 8 * sizeof(long long));

#if __INT_WIDTH__ == 32
  static_assert(__builtin_bit_cast(unsigned, __builtin_elementwise_popcount((vector4char){1, 2, 3, 4})) == (LITTLE_END ? 0x01020101 : 0x01010201));
#endif
}

namespace BuiltinMemcpy {
  constexpr int simple() {
    int a = 12;
    int b = 0;
    __builtin_memcpy(&b, &a, sizeof(a));
    return b;
  }
  static_assert(simple() == 12);

  constexpr bool arrayMemcpy() {
    char src[] = "abc";
    char dst[4] = {};
    __builtin_memcpy(dst, src, 4);
    return dst[0] == 'a' && dst[1] == 'b' && dst[2] == 'c' && dst[3] == '\0';
  }
  static_assert(arrayMemcpy());

  extern struct Incomplete incomplete;
  constexpr struct Incomplete *null_incomplete = 0;
  static_assert(__builtin_memcpy(null_incomplete, null_incomplete, sizeof(wchar_t))); // both-error {{not an integral constant expression}} \
                                                                                      // both-note {{source of 'memcpy' is nullptr}}

  wchar_t global;
  constexpr wchar_t *null = 0;
  static_assert(__builtin_memcpy(&global, null, sizeof(wchar_t))); // both-error {{not an integral constant expression}} \
                                                                   // both-note {{source of 'memcpy' is nullptr}}

  constexpr int simpleMove() {
    int a = 12;
    int b = 0;
    __builtin_memmove(&b, &a, sizeof(a));
    return b;
  }
  static_assert(simpleMove() == 12);

  constexpr int memcpyTypeRem() { // both-error {{never produces a constant expression}}
    int a = 12;
    int b = 0;
    __builtin_memmove(&b, &a, 1); // both-note {{'memmove' not supported: size to copy (1) is not a multiple of size of element type 'int'}} \
                                  // both-note {{not supported}}
    return b;
  }
  static_assert(memcpyTypeRem() == 12); // both-error {{not an integral constant expression}} \
                                        // both-note {{in call to}}

  template<typename T>
  constexpr T result(T (&arr)[4]) {
    return arr[0] * 1000 + arr[1] * 100 + arr[2] * 10 + arr[3];
  }

  constexpr int test_memcpy(int a, int b, int n) {
    int arr[4] = {1, 2, 3, 4};
    __builtin_memcpy(arr + a, arr + b, n); // both-note {{overlapping memory regions}}
    return result(arr);
  }

  static_assert(test_memcpy(1, 2, sizeof(int)) == 1334);
  static_assert(test_memcpy(0, 1, sizeof(int) * 2) == 2334); // both-error {{not an integral constant expression}} \
                                                             // both-note {{in call}}

  /// Both memcpy and memmove must support pointers.
  constexpr bool moveptr() {
    int a = 0;
    void *x = &a;
    void *z = nullptr;

    __builtin_memmove(&z, &x, sizeof(void*));
    return z == x;
  }
  static_assert(moveptr());

  constexpr bool cpyptr() {
    int a = 0;
    void *x = &a;
    void *z = nullptr;

    __builtin_memcpy(&z, &x, sizeof(void*));
    return z == x;
  }
  static_assert(cpyptr());

#ifndef __AVR__
  constexpr int test_memmove(int a, int b, int n) {
    int arr[4] = {1, 2, 3, 4};
    __builtin_memmove(arr + a, arr + b, n); // both-note {{destination is not a contiguous array of at least 3 elements of type 'int'}}
    return result(arr);
  }
  static_assert(test_memmove(2, 0, 12) == 4234); // both-error {{constant}} \
                                                 // both-note {{in call}}
#endif

  struct Trivial { char k; short s; constexpr bool ok() { return k == 3 && s == 4; } };
  constexpr bool test_trivial() {
    Trivial arr[3] = {{1, 2}, {3, 4}, {5, 6}};
    __builtin_memcpy(arr, arr+1, sizeof(Trivial));
    __builtin_memmove(arr+1, arr, 2 * sizeof(Trivial));

    return arr[0].ok() && arr[1].ok() && arr[2].ok();
  }
  static_assert(test_trivial());

  // Check that an incomplete array is rejected.
  constexpr int test_incomplete_array_type() { // both-error {{never produces a constant}}
    extern int arr[];
    __builtin_memmove(arr, arr, 4 * sizeof(arr[0]));
    // both-note@-1 2{{'memmove' not supported: source is not a contiguous array of at least 4 elements of type 'int'}}
    return arr[0] * 1000 + arr[1] * 100 + arr[2] * 10 + arr[3];
  }
  static_assert(test_incomplete_array_type() == 1234); // both-error {{constant}} both-note {{in call}}


  constexpr bool memmoveOverlapping() {
    char s1[] {1, 2, 3};
    __builtin_memmove(s1, s1 + 1, 2 * sizeof(char));
    // Now: 2, 3, 3
    bool Result1 = (s1[0] == 2 && s1[1] == 3 && s1[2]== 3);

    __builtin_memmove(s1 + 1, s1, 2 * sizeof(char));
    // Now: 2, 2, 3
    bool Result2 = (s1[0] == 2 && s1[1] == 2 && s1[2]== 3);

    return Result1 && Result2;
  }
  static_assert(memmoveOverlapping());

#define fold(x) (__builtin_constant_p(0) ? (x) : (x))
  static_assert(__builtin_memcpy(&global, fold((wchar_t*)123), sizeof(wchar_t))); // both-error {{not an integral constant expression}} \
                                                                                  // both-note {{source of 'memcpy' is (void *)123}}
  static_assert(__builtin_memcpy(fold(reinterpret_cast<wchar_t*>(123)), &global, sizeof(wchar_t))); // both-error {{not an integral constant expression}} \
                                                                                                    // both-note {{destination of 'memcpy' is (void *)123}}


  constexpr float type_pun(const unsigned &n) {
    float f = 0.0f;
    __builtin_memcpy(&f, &n, 4); // both-note {{cannot constant evaluate 'memcpy' from object of type 'const unsigned int' to object of type 'float'}}
    return f;
  }
  static_assert(type_pun(0x3f800000) == 1.0f); // both-error {{constant}} \
                                               // both-note {{in call}}

  struct Base { int a; };
  struct Derived : Base { int b; };
  constexpr int test_derived_to_base(int n) {
    Derived arr[2] = {1, 2, 3, 4};
    Base *p = &arr[0];
    Base *q = &arr[1];
    __builtin_memcpy(p, q, sizeof(Base) * n); // both-note {{source is not a contiguous array of at least 2 elements of type 'BuiltinMemcpy::Base'}}
    return arr[0].a * 1000 + arr[0].b * 100 + arr[1].a * 10 + arr[1].b;
  }
  static_assert(test_derived_to_base(0) == 1234);
  static_assert(test_derived_to_base(1) == 3234);
  static_assert(test_derived_to_base(2) == 3434); // both-error {{constant}} \
                                                  // both-note {{in call}}
}

namespace Memcmp {
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

  struct Bool3Tuple { bool bb[3]; };
  constexpr Bool3Tuple kb000100 = {{false, true, false}};
  static_assert(sizeof(bool) != 1u || __builtin_memcmp(ks00fe00, kb000100.bb, 1) == 0); // both-error {{constant}} \
                                                                                        // both-note {{not supported}}

  constexpr char a = 'a';
  constexpr char b = 'a';
  static_assert(__builtin_memcmp(&a, &b, 1) == 0);

  extern struct Incomplete incomplete;
  static_assert(__builtin_memcmp(&incomplete, "", 0u) == 0);
  static_assert(__builtin_memcmp("", &incomplete, 0u) == 0);
  static_assert(__builtin_memcmp(&incomplete, "", 1u) == 42); // both-error {{not an integral constant}} \
                                                              // both-note {{not supported}}
  static_assert(__builtin_memcmp("", &incomplete, 1u) == 42); // both-error {{not an integral constant}} \
                                                              // both-note {{not supported}}

  static_assert(__builtin_memcmp(u8"abab\0banana", u8"abab\0banana", 100) == 0); // both-error {{not an integral constant}} \
                                                                                 // both-note {{dereferenced one-past-the-end}}

  static_assert(__builtin_bcmp("abaa", "abba", 3) != 0);
  static_assert(__builtin_bcmp("abaa", "abba", 2) == 0);
  static_assert(__builtin_bcmp("a\203", "a", 2) != 0);
  static_assert(__builtin_bcmp("a\203", "a\003", 2) != 0);
  static_assert(__builtin_bcmp(0, 0, 0) == 0);
  static_assert(__builtin_bcmp("abab\0banana", "abab\0banana", 100) == 0); // both-error {{not an integral constant}}\
                                                                           // both-note {{dereferenced one-past-the-end}}
  static_assert(__builtin_bcmp("abab\0banana", "abab\0canada", 100) != 0); // FIXME: Should we reject this?
  static_assert(__builtin_bcmp("abab\0banana", "abab\0canada", 7) != 0);
  static_assert(__builtin_bcmp("abab\0banana", "abab\0canada", 6) != 0);
  static_assert(__builtin_bcmp("abab\0banana", "abab\0canada", 5) == 0);


  static_assert(__builtin_wmemcmp(L"abaa", L"abba", 3) == -1);
  static_assert(__builtin_wmemcmp(L"abaa", L"abba", 2) == 0);
  static_assert(__builtin_wmemcmp(0, 0, 0) == 0);
#if __WCHAR_WIDTH__ == 32
  static_assert(__builtin_wmemcmp(L"a\x83838383", L"aa", 2) ==
                (wchar_t)-1U >> 31);
#endif
  static_assert(__builtin_wmemcmp(L"abab\0banana", L"abab\0banana", 100) == 0); // both-error {{not an integral constant}} \
                                                                                // both-note {{dereferenced one-past-the-end}}
  static_assert(__builtin_wmemcmp(L"abab\0banana", L"abab\0canada", 100) == -1); // FIXME: Should we reject this?
  static_assert(__builtin_wmemcmp(L"abab\0banana", L"abab\0canada", 7) == -1);
  static_assert(__builtin_wmemcmp(L"abab\0banana", L"abab\0canada", 6) == -1);
  static_assert(__builtin_wmemcmp(L"abab\0banana", L"abab\0canada", 5) == 0);

#if __cplusplus >= 202002L
  constexpr bool f() {
    char *c = new char[12];
    c[0] = 'b';

    char n = 'a';
    bool b = __builtin_memcmp(c, &n, 1) == 0;

    delete[] c;
    return !b;
  }
  static_assert(f());
#endif

}

namespace Memchr {
  constexpr const char *kStr = "abca\xff\0d";
  constexpr char kFoo[] = {'f', 'o', 'o'};

  static_assert(__builtin_memchr(kStr, 'a', 0) == nullptr);
  static_assert(__builtin_memchr(kStr, 'a', 1) == kStr);
  static_assert(__builtin_memchr(kStr, '\0', 5) == nullptr);
  static_assert(__builtin_memchr(kStr, '\0', 6) == kStr + 5);
  static_assert(__builtin_memchr(kStr, '\xff', 8) == kStr + 4);
  static_assert(__builtin_memchr(kStr, '\xff' + 256, 8) == kStr + 4);
  static_assert(__builtin_memchr(kStr, '\xff' - 256, 8) == kStr + 4);
  static_assert(__builtin_memchr(kFoo, 'x', 3) == nullptr);
  static_assert(__builtin_memchr(kFoo, 'x', 4) == nullptr); // both-error {{not an integral constant}} \
                                                            // both-note {{dereferenced one-past-the-end}}
  static_assert(__builtin_memchr(nullptr, 'x', 3) == nullptr); // both-error {{not an integral constant}} \
                                                               // both-note {{dereferenced null}}
  static_assert(__builtin_memchr(nullptr, 'x', 0) == nullptr);


#if defined(CHAR8_T)
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
  static_assert(__builtin_memchr(kU8Foo, u8'x', 4) == nullptr); // both-error {{not an integral constant}} \
                                                                // both-note {{dereferenced one-past-the-end}}
  static_assert(__builtin_memchr(nullptr, u8'x', 3) == nullptr); // both-error {{not an integral constant}} \
                                                                 // both-note {{dereferenced null}}
  static_assert(__builtin_memchr(nullptr, u8'x', 0) == nullptr);
#endif

  extern struct Incomplete incomplete;
  static_assert(__builtin_memchr(&incomplete, 0, 0u) == nullptr);
  static_assert(__builtin_memchr(&incomplete, 0, 1u) == nullptr); // both-error {{not an integral constant}} \
                                                                  // both-note {{read of incomplete type 'struct Incomplete'}}

  const unsigned char &u1 = 0xf0;
  auto &&i1 = (const signed char []){-128};
  static_assert(__builtin_memchr(&u1, -(0x0f + 1), 1) == &u1);
  static_assert(__builtin_memchr(i1, 0x80, 1) == i1);

  enum class E : unsigned char {};
  struct EPair { E e, f; };
  constexpr EPair ee{E{240}};
  static_assert(__builtin_memchr(&ee.e, 240, 1) == &ee.e); // both-error {{constant}} \
                                                           // both-note {{not supported}}

  constexpr bool kBool[] = {false, true, false};
  constexpr const bool *const kBoolPastTheEndPtr = kBool + 3;
  static_assert(sizeof(bool) != 1u || __builtin_memchr(kBoolPastTheEndPtr - 3, 1, 99) == kBool + 1); // both-error {{constant}} \
                                                                                                     // both-note {{not supported}}
  static_assert(sizeof(bool) != 1u || __builtin_memchr(kBool + 1, 0, 99) == kBoolPastTheEndPtr - 1); // both-error {{constant}} \
                                                                                                     // both-note {{not supported}}
  static_assert(sizeof(bool) != 1u || __builtin_memchr(kBoolPastTheEndPtr - 3, -1, 3) == nullptr); // both-error {{constant}} \
                                                                                                   // both-note {{not supported}}
  static_assert(sizeof(bool) != 1u || __builtin_memchr(kBoolPastTheEndPtr, 0, 1) == nullptr); // both-error {{constant}} \
                                                                                              // both-note {{not supported}}

  static_assert(__builtin_char_memchr(kStr, 'a', 0) == nullptr);
  static_assert(__builtin_char_memchr(kStr, 'a', 1) == kStr);
  static_assert(__builtin_char_memchr(kStr, '\0', 5) == nullptr);
  static_assert(__builtin_char_memchr(kStr, '\0', 6) == kStr + 5);
  static_assert(__builtin_char_memchr(kStr, '\xff', 8) == kStr + 4);
  static_assert(__builtin_char_memchr(kStr, '\xff' + 256, 8) == kStr + 4);
  static_assert(__builtin_char_memchr(kStr, '\xff' - 256, 8) == kStr + 4);
  static_assert(__builtin_char_memchr(kFoo, 'x', 3) == nullptr);
  static_assert(__builtin_char_memchr(kFoo, 'x', 4) == nullptr); // both-error {{not an integral constant}} \
                                                                 // both-note {{dereferenced one-past-the-end}}
  static_assert(__builtin_char_memchr(nullptr, 'x', 3) == nullptr); // both-error {{not an integral constant}} \
                                                                    // both-note {{dereferenced null}}
  static_assert(__builtin_char_memchr(nullptr, 'x', 0) == nullptr);

  static_assert(*__builtin_char_memchr(kStr, '\xff', 8) == '\xff');
  constexpr bool char_memchr_mutable() {
    char buffer[] = "mutable";
    *__builtin_char_memchr(buffer, 't', 8) = 'r';
    *__builtin_char_memchr(buffer, 'm', 8) = 'd';
    return __builtin_strcmp(buffer, "durable") == 0;
  }
  static_assert(char_memchr_mutable());

  constexpr bool b = !memchr("hello", 'h', 3); // both-error {{constant expression}} \
                                               // both-note {{non-constexpr function 'memchr' cannot be used in a constant expression}}

  constexpr bool f() {
    const char *c = "abcdef";
    return __builtin_char_memchr(c + 1, 'f', 1) == nullptr;
  }
  static_assert(f());
}

namespace Strchr {
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
  static_assert(__builtin_strchr(kFoo, 'x') == nullptr); // both-error {{not an integral constant}} \
                                                         // both-note {{dereferenced one-past-the-end}}
  static_assert(__builtin_strchr(nullptr, 'x') == nullptr); // both-error {{not an integral constant}} \
                                                            // both-note {{dereferenced null}}

  constexpr bool a = !strchr("hello", 'h'); // both-error {{constant expression}} \
                                            // both-note {{non-constexpr function 'strchr' cannot be used in a constant expression}}
}

namespace WMemChr {
  constexpr const wchar_t *kStr = L"abca\xffff\0dL";
  constexpr wchar_t kFoo[] = {L'f', L'o', L'o'};

  static_assert(__builtin_wmemchr(kStr, L'a', 0) == nullptr);
  static_assert(__builtin_wmemchr(kStr, L'a', 1) == kStr);
  static_assert(__builtin_wmemchr(kStr, L'\0', 5) == nullptr);
  static_assert(__builtin_wmemchr(kStr, L'\0', 6) == kStr + 5);
  static_assert(__builtin_wmemchr(kStr, L'\xffff', 8) == kStr + 4);
  static_assert(__builtin_wmemchr(kFoo, L'x', 3) == nullptr);
  static_assert(__builtin_wmemchr(kFoo, L'x', 4) == nullptr); // both-error {{not an integral constant}} \
                                                              // both-note {{dereferenced one-past-the-end}}
  static_assert(__builtin_wmemchr(nullptr, L'x', 3) == nullptr); // both-error {{not an integral constant}} \
                                                                 // both-note {{dereferenced null}}
  static_assert(__builtin_wmemchr(nullptr, L'x', 0) == nullptr);

  constexpr bool b = !wmemchr(L"hello", L'h', 3); // both-error {{constant expression}} \
                                                  // both-note {{non-constexpr function 'wmemchr' cannot be used in a constant expression}}

  constexpr wchar_t kStr2[] = {L'f', L'o', L'\xffff', L'o'};
  static_assert(__builtin_wmemchr(kStr2, L'\xffff', 4) == kStr2 + 2);


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
  static_assert(__builtin_wcschr(kFoo, L'x') == nullptr); // both-error {{not an integral constant}} \
                                                          // both-note {{dereferenced one-past-the-end}}
  static_assert(__builtin_wcschr(nullptr, L'x') == nullptr); // both-error {{not an integral constant}} \
                                                             // both-note {{dereferenced null}}


  constexpr bool c = !wcschr(L"hello", L'h'); // both-error {{constant expression}} \
                                              // both-note {{non-constexpr function 'wcschr' cannot be used in a constant expression}}
}

namespace WMemCpy {
  template<typename T>
  constexpr T result(T (&arr)[4]) {
    return arr[0] * 1000 + arr[1] * 100 + arr[2] * 10 + arr[3];
  }
  constexpr int test_wmemcpy(int a, int b, int n) {
    wchar_t arr[4] = {1, 2, 3, 4};
    __builtin_wmemcpy(arr + a, arr + b, n);
    // both-note@-1 2{{overlapping memory regions}}
    // both-note@-2 {{source is not a contiguous array of at least 2 elements of type 'wchar_t'}}
    // both-note@-3 {{destination is not a contiguous array of at least 3 elements of type 'wchar_t'}}
    return result(arr);
  }
  static_assert(test_wmemcpy(1, 2, 1) == 1334);
  static_assert(test_wmemcpy(2, 1, 1) == 1224);
  static_assert(test_wmemcpy(0, 1, 2) == 2334); // both-error {{constant}} both-note {{in call}}
  static_assert(test_wmemcpy(1, 0, 2) == 1124); // both-error {{constant}} both-note {{in call}}
  static_assert(test_wmemcpy(1, 2, 1) == 1334);
  static_assert(test_wmemcpy(0, 3, 1) == 4234);
  static_assert(test_wmemcpy(0, 3, 2) == 4234); // both-error {{constant}} both-note {{in call}}
  static_assert(test_wmemcpy(2, 0, 3) == 4234); // both-error {{constant}} both-note {{in call}}

  wchar_t global;
  constexpr wchar_t *null = 0;
  static_assert(__builtin_wmemcpy(&global, null, sizeof(wchar_t))); // both-error {{}} \
                                                                    // both-note {{source of 'wmemcpy' is nullptr}}
  static_assert(__builtin_wmemcpy(null, &global, sizeof(wchar_t))); // both-error {{}} \
                                                                    // both-note {{destination of 'wmemcpy' is nullptr}}
}

namespace WMemMove {
  template<typename T>
  constexpr T result(T (&arr)[4]) {
    return arr[0] * 1000 + arr[1] * 100 + arr[2] * 10 + arr[3];
  }

  constexpr int test_wmemmove(int a, int b, int n) {
    wchar_t arr[4] = {1, 2, 3, 4};
    __builtin_wmemmove(arr + a, arr + b, n);
    // both-note@-1 {{source is not a contiguous array of at least 2 elements of type 'wchar_t'}}
    // both-note@-2 {{destination is not a contiguous array of at least 3 elements of type 'wchar_t'}}
    return result(arr);
  }

  static_assert(test_wmemmove(1, 2, 1) == 1334);
  static_assert(test_wmemmove(2, 1, 1) == 1224);
  static_assert(test_wmemmove(0, 1, 2) == 2334);
  static_assert(test_wmemmove(1, 0, 2) == 1124);
  static_assert(test_wmemmove(1, 2, 1) == 1334);
  static_assert(test_wmemmove(0, 3, 1) == 4234);
  static_assert(test_wmemmove(0, 3, 2) == 4234); // both-error {{constant}} both-note {{in call}}
  static_assert(test_wmemmove(2, 0, 3) == 4234); // both-error {{constant}} both-note {{in call}}

  wchar_t global;
  constexpr wchar_t *null = 0;
  static_assert(__builtin_wmemmove(&global, null, sizeof(wchar_t))); // both-error {{}} \
                                                                     // both-note {{source of 'wmemmove' is nullptr}}
  static_assert(__builtin_wmemmove(null, &global, sizeof(wchar_t))); // both-error {{}} \
                                                                     // both-note {{destination of 'wmemmove' is nullptr}}

  // Check that a pointer to an incomplete array is rejected.
  constexpr int test_address_of_incomplete_array_type() { // both-error {{never produces a constant}}
    extern int arr[];
    __builtin_memmove(&arr, &arr, 4 * sizeof(arr[0])); // both-note 2{{cannot constant evaluate 'memmove' between objects of incomplete type 'int[]'}}
    return arr[0] * 1000 + arr[1] * 100 + arr[2] * 10 + arr[3];
  }
  static_assert(test_address_of_incomplete_array_type() == 1234); // both-error {{constant}} \
                                                                  // both-note {{in call}}
}

namespace Invalid {
  constexpr int test() { // both-error {{never produces a constant expression}}
    __builtin_abort(); // both-note 2{{subexpression not valid in a constant expression}}
    return 0;
  }
  static_assert(test() == 0); // both-error {{not an integral constant expression}} \
                              // both-note {{in call to}}
}

#if __cplusplus >= 202002L
namespace WithinLifetime {
  constexpr int a = 10;
  static_assert(__builtin_is_within_lifetime(&a));

  consteval int IsActive(bool ReadB) {
    union {
      int a, b;
    } A;
    A.a = 10;
    if (ReadB)
      return __builtin_is_within_lifetime(&A.b);
    return __builtin_is_within_lifetime(&A.a);
  }
  static_assert(IsActive(false));
  static_assert(!IsActive(true));

  static_assert(__builtin_is_within_lifetime((void*)nullptr)); // both-error {{not an integral constant expression}} \
                                                               // both-note {{'__builtin_is_within_lifetime' cannot be called with a null pointer}}

  constexpr int i = 2;
  constexpr int arr[2]{};
  void f() {
    __builtin_is_within_lifetime(&i + 1); // both-error {{call to consteval function '__builtin_is_within_lifetime' is not a constant expression}} \
                                          // both-note {{'__builtin_is_within_lifetime' cannot be called with a one-past-the-end pointer}} \
                                          // both-warning {{expression result unused}}
    __builtin_is_within_lifetime(arr + 2); // both-error {{call to consteval function '__builtin_is_within_lifetime' is not a constant expression}} \
                                           // both-note {{'__builtin_is_within_lifetime' cannot be called with a one-past-the-end pointer}} \
                                           // both-warning {{expression result unused}}
  }


  constexpr bool self = __builtin_is_within_lifetime(&self); // both-error {{must be initialized by a constant expression}} \
                                                             // both-note {{'__builtin_is_within_lifetime' cannot be called with a pointer to an object whose lifetime has not yet begun}} \
                                                             // ref-error {{call to consteval function '__builtin_is_within_lifetime' is not a constant expression}} \
                                                             // ref-note {{initializer of 'self' is not a constant expression}} \
                                                             // ref-note {{declared here}}

  int nontCE(int p) { // both-note {{declared here}}
    return __builtin_is_within_lifetime(&p); // both-error {{call to consteval function}} \
                                             // both-note {{function parameter 'p' with unknown value cannot be used in a constant expression}}
  }


  struct XStd {
    consteval XStd() {
      __builtin_is_within_lifetime(this); // both-note {{cannot be called with a pointer to an object whose lifetime has not yet begun}}
    }
  } xstd; // both-error {{is not a constant expression}} \
          // both-note {{in call to}}

  /// FIXME: We do not have per-element lifetime information for primitive arrays.
  /// See https://github.com/llvm/llvm-project/issues/147528
  consteval bool test_dynamic(bool read_after_deallocate) {
    std::allocator<int> a;
    int* p = a.allocate(1); // expected-note 2{{allocation performed here was not deallocated}}
    // a.allocate starts the lifetime of an array,
    // the complete object of *p has started its lifetime
    if (__builtin_is_within_lifetime(p))
      return false;
    std::construct_at(p);
    if (!__builtin_is_within_lifetime(p))
      return false;
    std::destroy_at(p);
    if (__builtin_is_within_lifetime(p))
      return false;
    a.deallocate(p, 1);
    if (read_after_deallocate)
      __builtin_is_within_lifetime(p); // ref-note {{read of heap allocated object that has been deleted}}
    return true;
  }
  static_assert(test_dynamic(false)); // expected-error {{not an integral constant expression}}
  static_assert(test_dynamic(true)); // both-error {{not an integral constant expression}} \
                                     // ref-note {{in call to}}
}

#ifdef __SIZEOF_INT128__
namespace I128Mul {
  constexpr int mul() {
    __int128 A = 10;
    __int128 B = 10;
    __int128 R;
    __builtin_mul_overflow(A, B, &R);
    return 1;
  }
  static_assert(mul() == 1);
}
#endif

namespace InitParam {
  constexpr int foo(int a) {
      __builtin_mul_overflow(20, 10, &a);
      return a;
  }
  static_assert(foo(10) == 200);
}

#endif
