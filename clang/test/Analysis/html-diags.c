// RUN: rm -fR %t
// RUN: mkdir %t
// RUN: %clang_analyze_cc1 -analyzer-output=html -analyzer-checker=core -o %t %s
// RUN: ls %t | grep report

// D30406: Test new html-single-file output
// RUN: rm -fR %t
// RUN: mkdir %t
// RUN: %clang_analyze_cc1 -analyzer-output=html-single-file -analyzer-checker=core -o %t %s
// RUN: ls %t | grep report

// PR16547: Test relative paths
// RUN: cd %t
// RUN: %clang_analyze_cc1 -analyzer-output=html -analyzer-checker=core -o testrelative %s
// RUN: ls %t/testrelative | grep report

// Currently this test mainly checks that the HTML diagnostics doesn't crash
// when handling macros will calls with macros.  We should actually validate
// the output, but that requires being able to match against a specifically
// generate HTML file.

#define DEREF(p) *p = 0xDEADBEEF

void has_bug(int *p) {
  DEREF(p);
}

#define CALL_HAS_BUG(q) has_bug(q)

void test_call_macro(void) {
  CALL_HAS_BUG(0);
}
