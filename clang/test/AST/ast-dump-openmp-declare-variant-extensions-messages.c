// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify %s -x c++

int dummy(void) { return 1; }

#pragma omp declare variant(dummy) match(implementation={extension(match_any,match_all)}, device={kind(cpu, gpu)}) // expected-error {{only a single match extension allowed per OpenMP context selector}} expected-note {{the previous context property 'match_any' used here}} // expected-note {{the ignored property spans until here}}
int base1(void) { return 2; }

#pragma omp declare variant(dummy) match(implementation={extension(match_none,match_none)}, device={kind(gpu, fpga)}) // expected-warning {{the context property 'match_none' was used already in the same 'omp declare variant' directive; property ignored}} expected-note {{the previous context property 'match_none' used here}} expected-note {{the ignored property spans until here}}
int base2(void) { return 3; }

#pragma omp declare variant(dummy) match(implementation={vendor(pgi), extension(match_none,match_any)}, device={kind(cpu, gpu)}) // expected-error {{only a single match extension allowed per OpenMP context selector}} expected-note {{the previous context property 'match_none' used here}} // expected-note {{the ignored property spans until here}}
int base3(void) { return 4; }


int test(void) {
  return base1() + base2() + base3();
}
