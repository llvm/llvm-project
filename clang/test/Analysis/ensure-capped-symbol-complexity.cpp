// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection %s -verify

// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ConfigDumper 2>&1 | FileCheck %s --match-full-lines
// CHECK: max-symbol-complexity = 35

void clang_analyzer_dump(int v);

void pumpSymbolComplexity() {
  extern int *p;
  *p = (*p + 1) & 1023; //  2
  *p = (*p + 1) & 1023; //  4
  *p = (*p + 1) & 1023; //  6
  *p = (*p + 1) & 1023; //  8
  *p = (*p + 1) & 1023; // 10
  *p = (*p + 1) & 1023; // 12
  *p = (*p + 1) & 1023; // 14
  *p = (*p + 1) & 1023; // 16
  *p = (*p + 1) & 1023; // 18
  *p = (*p + 1) & 1023; // 20
  *p = (*p + 1) & 1023; // 22
  *p = (*p + 1) & 1023; // 24
  *p = (*p + 1) & 1023; // 26
  *p = (*p + 1) & 1023; // 28
  *p = (*p + 1) & 1023; // 30
  *p = (*p + 1) & 1023; // 32
  *p = (*p + 1) & 1023; // 34

  // The complexity of "*p" is below 35, so it's accurate.
  clang_analyzer_dump(*p);
  // expected-warning-re@-1 {{{{^\({34}reg}}}}

  // We would increase the complexity over the threshold, thus it'll get simplified.
  *p = (*p + 1) & 1023; // Would be 36, which is over 35.

  // This dump used to print a hugely complicated symbol, over 800 complexity, taking really long to simplify.
  clang_analyzer_dump(*p);
  // expected-warning-re@-1 {{{{^}}conj_${{[0-9]+}}{int, LC{{[0-9]+}}, S{{[0-9]+}}, #{{[0-9]+}}} [debug.ExprInspection]{{$}}}}
}

void hugelyOverComplicatedSymbol() {
#define TEN_TIMES(x) x x x x x x x x x x
#define HUNDRED_TIMES(x) TEN_TIMES(TEN_TIMES(x))
  extern int *p;
  HUNDRED_TIMES(*p = (*p + 1) & 1023;)
  HUNDRED_TIMES(*p = (*p + 1) & 1023;)
  HUNDRED_TIMES(*p = (*p + 1) & 1023;)
  HUNDRED_TIMES(*p = (*p + 1) & 1023;)

  // This dump used to print a hugely complicated symbol, over 800 complexity, taking really long to simplify.
  clang_analyzer_dump(*p);
  // expected-warning-re@-1 {{{{^}}((((((((conj_${{[0-9]+}}{int, LC{{[0-9]+}}, S{{[0-9]+}}, #{{[0-9]+}}}) + 1) & 1023) + 1) & 1023) + 1) & 1023) + 1) & 1023 [debug.ExprInspection]{{$}}}}
#undef HUNDRED_TIMES
#undef TEN_TIMES
}

typedef unsigned long long __attribute__((aligned((8)))) u64a;
u64a compress64(u64a x, u64a m) {
  if ((x & m) == 0)
      return 0;
  x &= m;
  u64a mk = ~m << 1;
  for (unsigned i = 0; i < 6; i++) {
    u64a mp = mk ^ (mk << 1);
    mp ^= mp << 2;
    mp ^= mp << 4;
    mp ^= mp << 8;
    mp ^= mp << 16;
    mp ^= mp << 32;
    u64a mv = mp & m;
    m = (m ^ mv) | (mv >> (1 << i));
    u64a t = x & mv;
    x = (x ^ t) | (t >> (1 << i));
    mk = mk & ~mp;
  }
  return x;
}
void storecompressed512_64bit(u64a *m, u64a *x) {
  u64a v[8] = {
    compress64(x[0], m[0]),
    compress64(x[1], m[1]),
    compress64(x[2], m[2]),
    compress64(x[3], m[3]),
    compress64(x[4], m[4]),
    compress64(x[5], m[5]),
    compress64(x[6], m[6]),
    compress64(x[7], m[7]),
  };
  (void)v;
}
