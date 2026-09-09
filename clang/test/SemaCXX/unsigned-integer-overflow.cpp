// Note: -triple x86_64-pc-linux-gnu is here to ensure the size of "uint32_t"
// and "uint64_t" are of the expected size. Ideally, we would included cstdint
// but it's not available in the tests.
// RUN: %clang_cc1 %s -verify=on -fsyntax-only -std=gnu++2a -triple x86_64-pc-linux-gnu -Wunsigned-integer-overflow
// RUN: %clang_cc1 %s -verify=on -fsyntax-only -std=gnu++2a -triple x86_64-pc-linux-gnu -Wunsigned-integer-overflow -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 %s -verify=on -fsyntax-only -std=gnu++2a -triple x86_64-pc-linux-gnu -Wunsigned-integer-overflow -fexperimental-overflow-behavior-types -DOBT
// RUN: %clang_cc1 %s -verify=off -fsyntax-only -std=gnu++2a -triple x86_64-pc-linux-gnu
// off-no-diagnostics

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

static_assert(sizeof(uint32_t) == 4, "uint32_t should be 4 bytes long");
static_assert(sizeof(uint64_t) == 8, "uint64_t should be 8 bytes long");

uint32_t a = 1024u * (1024 * 1024) * 6; // on-warning{{unsigned constant expression wraps around: 1073741824 * 6 is 6442450944, which does not fit in 'unsigned int' and becomes 2147483648}}
// Storing into a wider type does not help: the arithmetic is done in
// 'unsigned int' and only the wrapped result is converted.
uint64_t b = 1024u * (1024 * 1024) * 9; // on-warning{{wraps around: 1073741824 * 9 is 9663676416}}
uint32_t c = 0xFFFFFFFFu + 1u;          // on-warning{{wraps around: 4294967295 + 1 is 4294967296, which does not fit in 'unsigned int' and becomes 0}}
constexpr uint32_t kMiB = 1024u * 1024;
uint32_t d = kMiB * 1024u * 9;          // on-warning{{wraps around: 1073741824 * 9}}
uint64_t e = 0xFFFFFFFFFFFFFFFFull * 2u; // on-warning{{wraps around: 18446744073709551615 * 2 is 36893488147419103230, which does not fit in 'unsigned long long' and becomes 18446744073709551614}}
const uint32_t f = 1024u * 1024 * 1024 * 9;     // on-warning{{wraps around: 1073741824 * 9}}
static_assert(f == 1073741824u, "");

// Initializers that must be constant expressions (constexpr variables,
// enumerators, static constexpr members) are evaluated by a different path
// and are not diagnosed currently.
constexpr uint32_t notChecked1 = 1024u * 1024 * 1024 * 9;
enum : uint32_t { notChecked2 = 1024u * 1024 * 1024 * 9 };
void NotCheckedLocal() {
  constexpr uint32_t l = 1024u * 1024 * 1024 * 9;
  (void)l;
}

uint32_t ok1 = 1024u * 1024 * 1024;
uint64_t ok2 = uint64_t(1024) * 1024 * 1024 * 9;
uint64_t ok3 = 9ull * 1024 * 1024 * 1024;
// Only + and * are checked: wrapping subtraction, negation and shifts are
// idiomatic ways to build masks and maximum values.
uint32_t ok4 = 0u - 1;
uint32_t ok5 = -1u;
uint32_t ok6 = ~0u;
uint32_t ok7 = 1u << 31;
// Widening, then truncating explicitly, is the idiom for an intended wrap.
uint32_t ok8 = static_cast<uint32_t>(uint64_t(0xFFFFFFFFu) + 1u);
uint32_t Runtime(uint32_t v) { return v * 1024u * 1024u; }

// Diagnosed per instantiation, at the template's source location.
template <unsigned N> uint32_t Bytes() {
  return N * 1024u * 1024u; // on-warning{{wraps around: 5120000 * 1024}}
}
uint32_t t1 = Bytes<4>();
uint32_t t2 = Bytes<5000>();
template <unsigned N> struct Size {
  static constexpr uint32_t kBytes = N * 1024u * 1024u; // not diagnosed, see above
};
uint32_t t3 = Size<5000>::kBytes;

// Arithmetic inside a constexpr function is diagnosed when the function is
// evaluated with constants, at the function's own source location.
constexpr uint32_t Mul(uint32_t x) { return x * 16777619u; } // on-warning{{wraps around: 4294967295 * 16777619}}
uint32_t g = Mul(0xFFFFFFFFu) + 0u;

// Unlike code that is never instantiated, a branch that is merely unreachable
// is still evaluated.
uint32_t DeadBranch() {
  const int kib = -1;
  if (kib >= 0) {
    return uint32_t(kib) * 1024u; // on-warning{{wraps around: 4294967295 * 1024}}
  }
  return 0;
}

#ifdef OBT
// Annotating the wrapping shall silence the warning
typedef unsigned int __attribute__((overflow_behavior(wrap))) wrap_u32;
wrap_u32 w = wrap_u32(0xFFFFFFFFu) + 1u;
#endif

// The group name can be used for suppression, and the check is not performed
// where the warning is ignored.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsigned-integer-overflow"
uint32_t suppressed = 0xFFFFFFFFu + 1u;
#pragma clang diagnostic pop
uint32_t afterPop = 0xFFFFFFFFu + 1u; // on-warning{{wraps around: 4294967295 + 1}}

// Wide integers use the arbitrary-precision path in both evaluators.
typedef unsigned _BitInt(128) u128;
u128 big = ((u128)1 << 127) * 2u; // on-warning{{wraps around: 170141183460469231731687303715884105728 * 2 is 340282366920938463463374607431768211456, which does not fit in 'u128' (aka 'unsigned _BitInt(128)') and becomes 0}}

// Operands narrower than int are promoted to int first, so no unsigned
// wraparound is involved.
int promoted = (unsigned char)200 + (unsigned char)100;

// Compound assignments are not checked.
constexpr uint32_t Acc(uint32_t x) {
  x *= 16777619u;
  return x;
}
uint32_t acc = Acc(0xFFFFFFFFu) + 0u;
