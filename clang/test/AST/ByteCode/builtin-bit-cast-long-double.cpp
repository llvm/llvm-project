// RUN: %clang_cc1 -verify=ref,both -std=c++2a -fsyntax-only -triple x86_64-apple-macosx10.14.0 %s
// RUN: %clang_cc1 -verify=ref,both -std=c++2a -fsyntax-only -triple x86_64-apple-macosx10.14.0 %s -fno-signed-char
// RUN: %clang_cc1 -verify=ref,both -std=c++2a -fsyntax-only -triple aarch64_be-linux-gnu %s

// RUN: %clang_cc1 -verify=expected,both -std=c++2a -fsyntax-only -triple x86_64-apple-macosx10.14.0 %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -verify=expected,both -std=c++2a -fsyntax-only -triple x86_64-apple-macosx10.14.0 %s -fno-signed-char -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -verify=expected,both -std=c++2a -fsyntax-only -triple aarch64_be-linux-gnu %s -fexperimental-new-constant-interpreter

#if !__x86_64
// both-no-diagnostics
#endif


typedef decltype(nullptr) nullptr_t;
typedef __INTPTR_TYPE__ intptr_t;

static_assert(sizeof(int) == 4);
static_assert(sizeof(long long) == 8);

template <class To, class From>
constexpr To bit_cast(const From &from) {
  static_assert(sizeof(To) == sizeof(From));
  return __builtin_bit_cast(To, from);
}

template <class Intermediate, class Init>
constexpr bool check_round_trip(const Init &init) {
  return bit_cast<Init>(bit_cast<Intermediate>(init)) == init;
}

template <class Intermediate, class Init>
constexpr Init round_trip(const Init &init) {
  return bit_cast<Init>(bit_cast<Intermediate>(init));
}




namespace test_long_double {
#if __x86_64
/// FIXME: We could enable this, but since it aborts, it causes the usual mempory leak.
#if 0
constexpr __int128_t test_cast_to_int128 = bit_cast<__int128_t>((long double)0); // expected-error{{must be initialized by a constant expression}}\
                                                                                 // expected-note{{in call}}
#endif
constexpr long double ld = 3.1425926539;

struct bytes {
  unsigned char d[16];
};

static_assert(round_trip<bytes>(ld), "");

static_assert(round_trip<long double>(10.0L));

constexpr long double foo() {
  bytes A = __builtin_bit_cast(bytes, ld);
  long double ld = __builtin_bit_cast(long double, A);
  return ld;
}
static_assert(foo() == ld);

constexpr bool f(bool read_uninit) {
  bytes b = bit_cast<bytes>(ld);
  unsigned char ld_bytes[10] = {
    0x0,  0x48, 0x9f, 0x49, 0xf0,
    0x3c, 0x20, 0xc9, 0x0,  0x40,
  };

  for (int i = 0; i != 10; ++i)
    if (ld_bytes[i] != b.d[i])
      return false;

  if (read_uninit && b.d[10]) // both-note{{read of uninitialized object is not allowed in a constant expression}}
    return false;

  return true;
}

static_assert(f(/*read_uninit=*/false), "");
static_assert(f(/*read_uninit=*/true), ""); // both-error{{static assertion expression is not an integral constant expression}} \
                                            // both-note{{in call to 'f(true)'}}
constexpr bytes ld539 = {
  0x0, 0x0,  0x0,  0x0,
  0x0, 0x0,  0xc0, 0x86,
  0x8, 0x40, 0x0,  0x0,
  0x0, 0x0,  0x0,  0x0,
};
constexpr long double fivehundredandthirtynine = 539.0;
static_assert(bit_cast<long double>(ld539) == fivehundredandthirtynine, "");

struct LD {
  long double v;
};

constexpr LD ld2 = __builtin_bit_cast(LD, ld539.d);
constexpr long double five39 = __builtin_bit_cast(long double, ld539.d);
static_assert(ld2.v == five39);

#else
static_assert(round_trip<__int128_t>(34.0L));
#endif
}
