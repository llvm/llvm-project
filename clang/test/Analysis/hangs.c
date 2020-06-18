// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker core,debug.ExprInspection

// Stuff that used to hang.

extern void __assert_fail(__const char *__assertion, __const char *__file,
                          unsigned int __line, __const char *__function)
    __attribute__((__noreturn__));
#define assert(expr) \
  ((expr) ? (void)(0) : __assert_fail(#expr, __FILE__, __LINE__, __func__))

void clang_analyzer_eval(int);

int g();

int f(int y) {
  return y + g();
}

int produce_a_very_large_symbol(int x) {
  return f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(
             f(f(f(f(f(f(f(f(f(f(f(f(f(f(f(x))))))))))))))))))))))))))))))));
}

void produce_an_exponentially_exploding_symbol(int x, int y) {
  x += y; y += x + g();
  x += y; y += x + g();
  x += y; y += x + g();
  x += y; y += x + g();
  x += y; y += x + g();
  x += y; y += x + g();
  x += y; y += x + g();
  x += y; y += x + g();
  x += y; y += x + g();
  x += y; y += x + g();
  x += y; y += x + g();
}

void produce_an_exponentially_exploding_symbol_2(int x, int y) {
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  if (x > 1) {
    if (x > 2) {
      if (x > 3) {
        if (x > 4) {
          if (x > 5) {
            if (x > 6) {
              if (x > 7) {
                if (x > 8) {
                  if (x > 9) {
                    if (x > 10) {
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void produce_an_exponentially_exploding_symbol_3(int x, int y) {
  assert(0 < x && x < 10);
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  x &= y;
  y &= x & g();
  clang_analyzer_eval(0 < x && x < 10); // expected-warning{{TRUE}}
                                        // expected-warning@-1{{FALSE}}
}
