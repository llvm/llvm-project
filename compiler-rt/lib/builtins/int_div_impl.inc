#define clz(a) (sizeof(a) == 8 ? __builtin_clzll(a) : __builtin_clz(a))

// Adapted from Figure 3-40 of The PowerPC Compiler Writer's Guide
static __inline fixuint_t __udivXi3(fixuint_t n, fixuint_t d) {
  const unsigned N = sizeof(fixuint_t) * CHAR_BIT;
  // d == 0 cases are unspecified.
  unsigned sr = (d ? clz(d) : N) - (n ? clz(n) : N);
  // 0 <= sr <= N - 1 or sr is very large.
  if (sr > N - 1) // n < d
    return 0;
  if (sr == N - 1) // d == 1
    return n;
  ++sr;
  // 1 <= sr <= N - 1. Shifts do not trigger UB.
  fixuint_t r = n >> sr;
  n <<= N - sr;
  fixuint_t carry = 0;
  for (; sr > 0; --sr) {
    r = (r << 1) | (n >> (N - 1));
    n = (n << 1) | carry;
    // Branch-less version of:
    // carry = 0;
    // if (r >= d) r -= d, carry = 1;
    const fixint_t s = (fixint_t)(d - r - 1) >> (N - 1);
    carry = s & 1;
    r -= d & s;
  }
  n = (n << 1) | carry;
  return n;
}

// Mostly identical to __udivXi3 but the return values are different.
static __inline fixuint_t __umodXi3(fixuint_t n, fixuint_t d) {
  const unsigned N = sizeof(fixuint_t) * CHAR_BIT;
  // d == 0 cases are unspecified.
  unsigned sr = (d ? clz(d) : N) - (n ? clz(n) : N);
  // 0 <= sr <= N - 1 or sr is very large.
  if (sr > N - 1) // n < d
    return n;
  if (sr == N - 1) // d == 1
    return 0;
  ++sr;
  // 1 <= sr <= N - 1. Shifts do not trigger UB.
  fixuint_t r = n >> sr;
  n <<= N - sr;
  fixuint_t carry = 0;
  for (; sr > 0; --sr) {
    r = (r << 1) | (n >> (N - 1));
    n = (n << 1) | carry;
    // Branch-less version of:
    // carry = 0;
    // if (r >= d) r -= d, carry = 1;
    const fixint_t s = (fixint_t)(d - r - 1) >> (N - 1);
    carry = s & 1;
    r -= d & s;
  }
  return r;
}
