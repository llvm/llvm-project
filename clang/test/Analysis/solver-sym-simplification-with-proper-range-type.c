// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -verify

// Here we test that the range based solver equivalency tracking mechanism
// assigns a properly typed range to the simplified symbol.

void clang_analyzer_printState(void);
void clang_analyzer_eval(int);

void f(int a0, int b0, int c)
{
    int a1 = a0 - b0;
    int b1 = (unsigned)a1 + c;
    if (c == 0) {

        int d = 7L / b1; // ...
        // At this point b1 is considered non-zero, which results in a new
        // constraint for $a0 - $b0 + $c. The type of this sym is unsigned,
        // however, the simplified sym is $a0 - $b0 and its type is signed.
        // This is probably the result of the inherent improper handling of
        // casts. Anyway, Range assignment for constraints use this type
        // information. Therefore, we must make sure that first we simplify the
        // symbol and only then we assign the range.

        clang_analyzer_eval(a0 - b0 != 0); // expected-warning{{TRUE}}
    }
}
