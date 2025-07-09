// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -std=c++11 -verify=expected,nosink -analyzer-config eagerly-assume=false %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,debug.ExprInspection -std=c++11 -verify=expected,sink -analyzer-config eagerly-assume=false %s

// This test validates that calling an unknown destructor invalidates the
// members of an object. This was originally a part of the test file `new.cpp`,
// but was split off into a separate file because the checker family
// implemented in `MallocChecker.cpp` (which is activated via unix.Malloc in
// `new.cpp` sinks all execution paths that refer to members of a deleted object.

void clang_analyzer_eval(bool);

// Invalidate Region even in case of default destructor
class InvalidateDestTest {
public:
  int x;
  int *y;
  ~InvalidateDestTest();
};

int test_member_invalidation() {

  //test invalidation of member variable
  InvalidateDestTest *test = new InvalidateDestTest();
  test->x = 5;
  int *k = &(test->x);
  clang_analyzer_eval(*k == 5); // expected-warning{{TRUE}}
  delete test;
  clang_analyzer_eval(*k == 5); // nosink-warning{{UNKNOWN}}

  //test invalidation of member pointer
  int localVar = 5;
  test = new InvalidateDestTest();
  test->y = &localVar;
  delete test;
  clang_analyzer_eval(localVar == 5); // nosink-warning{{UNKNOWN}}

  // Test aray elements are invalidated.
  int Var1 = 5;
  int Var2 = 5;
  InvalidateDestTest *a = new InvalidateDestTest[2];
  a[0].y = &Var1;
  a[1].y = &Var2;
  delete[] a;
  clang_analyzer_eval(Var1 == 5); // nosink-warning{{UNKNOWN}}
  clang_analyzer_eval(Var2 == 5); // nosink-warning{{UNKNOWN}}
  return 0;
}
