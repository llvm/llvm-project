// Note: -triple x86_64-pc-linux-gnu is here to ensure the size of "uint32_t"
// and "uint64_t" are of the expected size. Ideally, we would included stdint.h
// but it's not available in the tests.
// RUN: %clang_cc1 %s -verify=on -fsyntax-only -triple x86_64-pc-linux-gnu -Wunsigned-integer-overflow
// RUN: %clang_cc1 %s -verify=on -fsyntax-only -triple x86_64-pc-linux-gnu -Wunsigned-integer-overflow -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 %s -verify=off -fsyntax-only -triple x86_64-pc-linux-gnu
// off-no-diagnostics

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

_Static_assert(sizeof(uint32_t) == 4, "uint32_t should be 4 bytes long");
_Static_assert(sizeof(uint64_t) == 8, "uint64_t should be 8 bytes long");

uint32_t a = 1024u * (1024 * 1024) * 6; // on-warning{{unsigned constant expression wraps around: 1073741824 * 6 is 6442450944, which does not fit in 'unsigned int' and becomes 2147483648}}
uint64_t b = 1024u * (1024 * 1024) * 9; // on-warning{{wraps around: 1073741824 * 9 is 9663676416}}
uint32_t c = 0xFFFFFFFFu + 1u;          // on-warning{{wraps around: 4294967295 + 1 is 4294967296}}
enum { kMiB = 1024 * 1024 };
uint32_t d = kMiB * 1024u * 9;          // on-warning{{wraps around: 1073741824 * 9}}

uint32_t ok1 = 1024u * 1024 * 1024;
uint64_t ok2 = (uint64_t)1024 * 1024 * 1024 * 9;
uint32_t ok3 = 0u - 1;
uint32_t ok4 = ~0u;
uint32_t ok5 = (uint32_t)((uint64_t)0xFFFFFFFFu + 1u);
uint32_t Runtime(uint32_t v) { return v * 1024u * 1024u; }

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
