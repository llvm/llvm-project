// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -verify

// Here, we test that symbol simplification in the solver does not produce any
// crashes.

// expected-no-diagnostics

static int a, b;
static long c;

static void f(int i, int j)
{
    (void)(j <= 0 && i ? i : j);
}

static void g(void)
{
    int d = a - b | (c < 0);
    for (;;)
    {
        f(d ^ c, c);
    }
}
