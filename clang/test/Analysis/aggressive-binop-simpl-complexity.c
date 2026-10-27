// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config aggressive-binary-operation-simplification=true -verify %s

// Verify that symbols created by the 'tryRearrange' logic respect the symbol
// complexity threshold. This is adapted from the older test PR38208.c to show
// a clear failure instead of a hanging analysis (exponential runtime).

void clang_analyzer_dump(int);

int foo(int x, int y) {
  int a = x; int b = y; // complexity: 1, 1
  a += b; b -= a; // complexity: 2, 3
  a += b; b -= a; // complexity: 5, 8
  a += b; b -= a; // complexity: 13, 21
  a += b; b -= a; // complexity: 34, would be 55
 
  // We assume concrete values for 'x' and 'y' only after the calculations, to
  // ensure that the calculations were performed symbolically.
  if (x == 0 && y == 0) {
    // The symbolic expression for 'a' has complexity 34, which is not above
    // the threshold (35), so the analyzer can remember the connection between
    // 'a' and the arguments 'x' and 'y'.
    clang_analyzer_dump(a); //expected-warning {{0 S32}}
    // The symbolic expression for 'b' would have complexity 55, which is above
    // the threshold (35), so the analyzer conjured a fresh symbol, which is
    // totally unrelated to 'x' and 'y'.
    clang_analyzer_dump(b); //expected-warning {{conj_}}
  }
  return a;
}
