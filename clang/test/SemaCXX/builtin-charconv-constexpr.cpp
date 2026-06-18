// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// expected-no-diagnostics

// POC scope: the conversion is implemented in the classic constant evaluator
// (ExprConstant.cpp). The experimental bytecode interpreter
// (-fexperimental-new-constant-interpreter) would need a parallel
// implementation in clang/lib/AST/ByteCode and is left as follow-up.

// POC: __builtin_to_chars / __builtin_from_chars are usable in constant
// expressions, for integers of any width including _BitInt(N > 128).

template <class T> constexpr bool rt(T v, int base) {
  char buf[2100] = {};
  char *end = __builtin_to_chars(buf, buf + sizeof(buf), v, base);
  if (!end)
    return false;
  T out = 0;
  int ec = 0;
  const char *p = __builtin_from_chars(buf, end, &out, base, &ec);
  return ec == 0 && p == end && out == v;
}

template <class T> constexpr bool rt_all_bases(T v) {
  return rt(v, 2) && rt(v, 8) && rt(v, 10) && rt(v, 16) && rt(v, 36) &&
         rt(v, 3) && rt(v, 7);
}

// Standard integer types, including edge values.
static_assert(rt_all_bases<int>(0));
static_assert(rt_all_bases<int>(1));
static_assert(rt_all_bases<int>(-1));
static_assert(rt_all_bases<int>(2147483647));
static_assert(rt_all_bases<int>(-2147483647 - 1)); // most-negative
static_assert(rt_all_bases<unsigned>(0u));
static_assert(rt_all_bases<unsigned>(4294967295u));
static_assert(rt_all_bases<long long>(-9223372036854775807LL - 1));
static_assert(rt_all_bases<unsigned long long>(18446744073709551615ULL));

#ifdef __SIZEOF_INT128__
static_assert(rt_all_bases<__int128>(0));
static_assert(rt_all_bases<__int128>((__int128)-1));
static_assert(rt_all_bases<unsigned __int128>(~(unsigned __int128)0));
#endif

// _BitInt wider than 128 bits: the motivating case.
using s256 = signed _BitInt(256);
using u256 = unsigned _BitInt(256);
using s1024 = signed _BitInt(1024);

constexpr s256 mk256() {
  s256 v = 1;
  for (int i = 0; i < 150; ++i) // 3^150 < 2^255, stays in range
    v = v * 3 + 1;
  return v;
}

static_assert(rt_all_bases<u256>(0));
static_assert(rt_all_bases<u256>(~(u256)0)); // all ones
static_assert(rt_all_bases<s256>(mk256()));
static_assert(rt_all_bases<s256>(-mk256()));
// Most-negative via unsigned cast; a signed shift into the sign bit is UB.
static_assert(rt_all_bases<s256>((s256)(((u256)1) << 255)));
static_assert(rt(((s1024)1) << 1000, 10));
static_assert(rt(((u256)1) << 200, 16));

// Buffer too small -> null return.
constexpr bool too_small() {
  char buf[2] = {};
  return __builtin_to_chars(buf, buf + 2, 12345, 10) == nullptr;
}
static_assert(too_small());

// Exact-fit buffer succeeds.
constexpr bool exact_fit() {
  char buf[3] = {};
  char *end = __builtin_to_chars(buf, buf + 3, -42, 10);
  return end == buf + 3 && buf[0] == '-' && buf[1] == '4' && buf[2] == '2';
}
static_assert(exact_fit());

// from_chars error classes and parsing edges.
constexpr int parse_ec(const char *s, unsigned long n, int base) {
  long long out = 0;
  int ec = 0;
  __builtin_from_chars(s, s + n, &out, base, &ec);
  return ec;
}
static_assert(parse_ec("", 0, 10) == 1);         // invalid_argument
static_assert(parse_ec("xyz", 3, 10) == 1);      // invalid_argument
static_assert(parse_ec("+5", 2, 10) == 1);       // '+' not accepted
static_assert(parse_ec("123", 3, 10) == 0);      // ok
static_assert(parse_ec("99999999999999999999999999999", 29, 10) == 2); // range

// from_chars stops at the first non-digit; no base prefix is consumed.
constexpr bool partial() {
  long long out = 0;
  int ec = 0;
  const char *s = "0x1f";
  const char *p = __builtin_from_chars(s, s + 4, &out, 16, &ec);
  return ec == 0 && out == 0 && p == s + 1; // parsed only "0"
}
static_assert(partial());

// '8' is not a base-8 digit; case-insensitive letters in base 16.
static_assert(parse_ec("8", 1, 8) == 1);
constexpr bool hex_case() {
  const char upper[] = "FF";
  const char lower[] = "ff";
  long long a = 0, b = 0;
  int ea = 0, eb = 0;
  __builtin_from_chars(upper, upper + 2, &a, 16, &ea);
  __builtin_from_chars(lower, lower + 2, &b, 16, &eb);
  return ea == 0 && eb == 0 && a == 255 && b == 255;
}
static_assert(hex_case());

// Independent-oracle anchors. The round-trip checks above pass even if a
// symmetric bug hits both directions; these compare to_chars output and
// from_chars values against fixed expected results, per base. INV-1 of the
// review establishes the runtime engine matches the constexpr engine
// byte-for-byte, so these constexpr anchors cover the runtime path too.
constexpr bool to_eq(auto v, int base, const char *expect) {
  char buf[300] = {};
  char *e = __builtin_to_chars(buf, buf + sizeof(buf), v, base);
  if (!e)
    return false;
  const char *p = buf;
  for (; *expect; ++p, ++expect)
    if (p == e || *p != *expect)
      return false;
  return p == e;
}
static_assert(to_eq(5u, 2, "101"));
static_assert(to_eq(8u, 8, "10"));
static_assert(to_eq(255u, 16, "ff"));
static_assert(to_eq(35u, 36, "z"));
static_assert(to_eq(1000000u, 10, "1000000"));
static_assert(to_eq(-255, 16, "-ff"));
static_assert(to_eq((s256)(((u256)1) << 200), 10,
                    "16069380442589902755419620923411626025222029937827928353"
                    "01376"));

constexpr long long parse_val(const char *s, unsigned n, int base) {
  long long out = -999;
  int ec = 0;
  __builtin_from_chars(s, s + n, &out, base, &ec);
  return out;
}
static_assert(parse_val("101", 3, 2) == 5);
static_assert(parse_val("777", 3, 8) == 511);
static_assert(parse_val("7f", 2, 16) == 127);
static_assert(parse_val("z", 1, 36) == 35);
static_assert(parse_val("-123", 4, 10) == -123);
