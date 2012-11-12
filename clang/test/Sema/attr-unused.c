// RUN: %clang_cc1 -verify -Wunused -Wused-but-marked-unused -Wunused-parameter -Wunused -fsyntax-only %s

static void (*fp0)(void) __attribute__((unused));

static void __attribute__((unused)) f0(void);

// On K&R
int f1() __attribute__((unused));

int g0 __attribute__((unused));

int f2() __attribute__((unused(1, 2))); // expected-error {{attribute takes no arguments}}

struct Test0_unused {} __attribute__((unused));
struct Test0_not_unused {};
typedef int Int_unused __attribute__((unused));
typedef int Int_not_unused;

void test0() {
  int x; // expected-warning {{unused variable}}

  Int_not_unused i0; // expected-warning {{unused variable}}
  Int_unused i1; // expected-warning {{'Int_unused' was marked unused but was used}}

  struct Test0_not_unused s0; // expected-warning {{unused variable}}
  struct Test0_unused s1; // expected-warning {{'Test0_unused' was marked unused but was used}}
}

int f3(int x) { // expected-warning{{unused parameter 'x'}}
  return 0;
}

int f4(int x) {
  return x;
}

int f5(int x __attribute__((__unused__))) {
  return 0;
}

int f6(int x __attribute__((__unused__))) {
  return x; // expected-warning{{'x' was marked unused but was used}}
}
