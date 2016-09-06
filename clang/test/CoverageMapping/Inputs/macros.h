// Assorted macros to help test #include behavior across file boundaries.

#define helper1 0

void helper2(const char *, ...);

#define M1(a, ...) helper2(a, ##__VA_ARGS__);

// Note: M2 stresses vararg macro functions with macro arguments. The spelling
// locations of the args used to be set to the expansion site, leading to
// crashes (region LineEnd < LineStart). The regression test requires M2's line
// number to be greater than the line number containing the expansion.
#define M2(a, ...) M1(a, helper1, ##__VA_ARGS__);
