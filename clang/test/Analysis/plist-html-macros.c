// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s
// (basic correctness check)

// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
//
// RUN: %clang_analyze_cc1 -o %t.dir/index.plist %s \
// RUN:   -analyzer-checker=core -analyzer-output=plist-html
//
// RUN: ls %t.dir | grep '\.html' | count 1
// RUN: grep '\.html' %t.dir/index.plist | count 1

// This tests two things: that the two calls to null_deref below are coalesced
// into a single bug by both the plist and HTML diagnostics, and that the plist
// diagnostics have a reference to the HTML diagnostics. (It would be nice to
// check more carefully that the two actually match, but that's hard to write
// in a lit RUN line.)

#define CALL_FN(a) null_deref(a)

void null_deref(int *a) {
  if (a)
    return;
  *a = 1; // expected-warning{{null}}
}

void test1() {
  CALL_FN(0);
}

void test2(int *p) {
  CALL_FN(p);
}
