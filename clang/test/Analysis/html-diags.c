// RUN: mkdir %T/dir
// RUN: %clang_cc1 -analyze -analyzer-output=html -analyzer-checker=core -o %T/dir %s
// RUN: rm -fR %T/dir

// Currently this test mainly checks that the HTML diagnostics doesn't crash
// when handling macros will calls with macros.  We should actually validate
// the output, but that requires being able to match against a specifically
// generate HTML file.

#define DEREF(p) *p = 0xDEADBEEF

void has_bug(int *p) {
  DEREF(p);
}

#define CALL_HAS_BUG(q) has_bug(q)

void test_call_macro() {
  CALL_HAS_BUG(0);
}
