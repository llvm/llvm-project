// RUN: %clang_cc1 -verify %s

#define P_10(x) x##0, x##1, x##2, x##3, x##4, x##5, x##6, x##7, x##8, x##9,
#define P_100(x) P_10(x##0) P_10(x##1) P_10(x##2) P_10(x##3) P_10(x##4) \
                 P_10(x##5) P_10(x##6) P_10(x##7) P_10(x##8) P_10(x##9)
#define P_1000(x) P_100(x##0) P_100(x##1) P_100(x##2) P_100(x##3) P_100(x##4) \
                  P_100(x##5) P_100(x##6) P_100(x##7) P_100(x##8) P_100(x##9)
#define P_10000(x) P_1000(x##0) P_1000(x##1) P_1000(x##2) P_1000(x##3) P_1000(x##4) \
                   P_1000(x##5) P_1000(x##6) P_1000(x##7) P_1000(x##8) P_1000(x##9)

void func (
  P_10000(int p)
  P_10000(int q)
  P_10000(int r)
  P_10000(int s)
  P_10000(int t)
  P_10000(int u)
  P_10000(int v) // expected-error {{too many function parameters; subsequent parameters will be ignored}}
  int w);

extern double(*func2)(
  P_10000(int p)
  P_10000(int q)
  P_10000(int r)
  P_10000(int s)
  P_10000(int t)
  P_10000(int u)
  P_10000(int v) // expected-error {{too many function parameters; subsequent parameters will be ignored}}
  int w);
