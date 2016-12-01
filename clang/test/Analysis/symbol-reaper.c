// RUN: %clang_cc1 -analyze -analyzer-checker=debug.ExprInspection -verify %s

void clang_analyzer_eval(int);
void clang_analyzer_warnOnDeadSymbol(int);
void clang_analyzer_numTimesReached();

int conjure_index();

void test_that_expr_inspection_works() {
  do {
    int x = conjure_index();
    clang_analyzer_warnOnDeadSymbol(x);
  } while(0); // expected-warning{{SYMBOL DEAD}}

  // Make sure we don't accidentally split state in ExprInspection.
  clang_analyzer_numTimesReached(); // expected-warning{{1}}
}

// These tests verify the reaping of symbols that are only referenced as
// index values in element regions. Most of the time, depending on where
// the element region, as Loc value, is stored, it is possible to
// recover the index symbol in checker code, which is also demonstrated
// in the return_ptr_range.c test file.

int arr[3];

int *test_element_index_lifetime_in_environment_values() {
  int *ptr;
  do {
    int x = conjure_index();
    clang_analyzer_warnOnDeadSymbol(x);
    ptr = arr + x;
  } while (0);
  return ptr;
}

void test_element_index_lifetime_in_store_keys() {
  do {
    int x = conjure_index();
    clang_analyzer_warnOnDeadSymbol(x);
    arr[x] = 1;
    if (x) {}
  } while (0); // no-warning
}

int *ptr;
void test_element_index_lifetime_in_store_values() {
  do {
    int x = conjure_index();
    clang_analyzer_warnOnDeadSymbol(x);
    ptr = arr + x;
  } while (0); // no-warning
}

struct S1 {
  int field;
};
struct S2 {
  struct S1 array[5];
} s2;

void test_element_index_lifetime_with_complicated_hierarchy_of_regions() {
  do {
    int x = conjure_index();
    clang_analyzer_warnOnDeadSymbol(x);
    s2.array[x].field = 1;
    if (x) {}
  } while (0); // no-warning
}

// Test below checks lifetime of SymbolRegionValue in certain conditions.

int **ptrptr;
void test_region_lifetime_as_store_value(int *x) {
  clang_analyzer_warnOnDeadSymbol((int) x);
  *x = 1;
  ptrptr = &x;
  (void)0; // No-op; make sure the environment forgets things and the GC runs.
  clang_analyzer_eval(**ptrptr); // expected-warning{{TRUE}}
} // no-warning
