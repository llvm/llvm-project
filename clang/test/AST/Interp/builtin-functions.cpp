// RUN: %clang_cc1 -Wno-string-plus-int -fexperimental-new-constant-interpreter %s -verify=expected,both
// RUN: %clang_cc1 -Wno-string-plus-int -fexperimental-new-constant-interpreter -triple i686 %s -verify=expected,both
// RUN: %clang_cc1 -Wno-string-plus-int -verify=ref,both %s -Wno-constant-evaluated
// RUN: %clang_cc1 -std=c++20 -Wno-string-plus-int -fexperimental-new-constant-interpreter %s -verify=expected,both
// RUN: %clang_cc1 -std=c++20 -Wno-string-plus-int -fexperimental-new-constant-interpreter -triple i686 %s -verify=expected,both
// RUN: %clang_cc1 -std=c++20 -Wno-string-plus-int -verify=ref,both %s -Wno-constant-evaluated
// RUN: %clang_cc1 -triple avr -std=c++20 -Wno-string-plus-int -fexperimental-new-constant-interpreter %s -verify=expected,both
// RUN: %clang_cc1 -triple avr -std=c++20 -Wno-string-plus-int -verify=ref,both %s -Wno-constant-evaluated


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
                                                       // both-note {{dereferenced null}} \
                                                       // expected-note {{in call to}}
  static_assert(__builtin_strcmp("abab", 0) == 0, ""); // both-error {{not an integral constant}} \
                                                       // both-note {{dereferenced null}} \
                                                       // expected-note {{in call to}}

  static_assert(__builtin_strcmp(kFoobar, kFoobazfoobar) == -1, "");
  static_assert(__builtin_strcmp(kFoobar, kFoobazfoobar + 6) == 0, ""); // both-error {{not an integral constant}} \
                                                                        // both-note {{dereferenced one-past-the-end}} \
                                                                        // expected-note {{in call to}}

  /// Used to assert because we're passing a dummy pointer to
  /// __builtin_strcmp() when evaluating the return statement.
  constexpr bool char_memchr_mutable() {
    char buffer[] = "mutable";
    return __builtin_strcmp(buffer, "mutable") == 0;
  }
  static_assert(char_memchr_mutable(), "");
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
                                                 // both-note {{one-past-the-end}} \
                                                 // expected-note {{in call to}}
  constexpr int over2 = __builtin_strlen(b + 9); // both-error {{constant expression}} \
                                                 // both-note {{one-past-the-end}} \
                                                 // expected-note {{in call to}}
  constexpr int over3 = __builtin_strlen(c + 9); // both-error {{constant expression}} \
                                                 // both-note {{one-past-the-end}} \
                                                 // expected-note {{in call to}}

  constexpr int under1 = __builtin_strlen(a - 1); // both-error {{constant expression}} \
                                                  // both-note {{cannot refer to element -1}}
  constexpr int under2 = __builtin_strlen(b - 1); // both-error {{constant expression}} \
                                                  // both-note {{cannot refer to element -1}}
  constexpr int under3 = __builtin_strlen(c - 1); // both-error {{constant expression}} \
                                                  // both-note {{cannot refer to element -1}}

  constexpr char d[] = { 'f', 'o', 'o' }; // no nul terminator.
  constexpr int bad = __builtin_strlen(d); // both-error {{constant expression}} \
                                           // both-note {{one-past-the-end}} \
                                           // expected-note {{in call to}}
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
  //constexpr long double NaN5 = __builtin_nanf128("");

  /// FIXME: This should be accepted by the current interpreter as well.
  constexpr char f[] = {'0', 'x', 'A', 'E', '\0'};
  constexpr double NaN6 = __builtin_nan(f); // ref-error {{must be initialized by a constant expression}}

  /// FIXME: Current interpreter misses diagnostics.
  constexpr char f2[] = {'0', 'x', 'A', 'E'}; /// No trailing 0 byte.
  constexpr double NaN7 = __builtin_nan(f2); // both-error {{must be initialized by a constant expression}} \
                                             // expected-note {{read of dereferenced one-past-the-end pointer}} \
                                             // expected-note {{in call to}}
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
}

namespace ctz {
  char ctz1[__builtin_ctz(1) == 0 ? 1 : -1];
  char ctz2[__builtin_ctz(8) == 3 ? 1 : -1];
  char ctz3[__builtin_ctz(1 << (BITSIZE(int) - 1)) == BITSIZE(int) - 1 ? 1 : -1];
  int ctz4 = __builtin_ctz(0);
  char ctz5[__builtin_ctzl(0x10L) == 4 ? 1 : -1];
  char ctz6[__builtin_ctzll(0x100LL) == 8 ? 1 : -1];
  char ctz7[__builtin_ctzs(1 << (BITSIZE(short) - 1)) == BITSIZE(short) - 1 ? 1 : -1];
}

namespace bswap {
  extern int f(void);
  int h3 = __builtin_bswap16(0x1234) == 0x3412 ? 1 : f();
  int h4 = __builtin_bswap32(0x1234) == 0x34120000 ? 1 : f();
  int h5 = __builtin_bswap64(0x1234) == 0x3412000000000000 ? 1 : f();
}
