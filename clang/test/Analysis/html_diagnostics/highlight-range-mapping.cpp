// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: %clang_analyze_cc1 -analyzer-checker=core,deadcode.DeadStores \
// RUN:   -analyzer-output=html -o %t -std=c++17 -verify %s
// RUN: cat %t/report-*.html | FileCheck %s

// CHECK-DAG is required: one report file is emitted per diagnostic, and their
// names are content hashes, so the glob order is not the source order.

// expected-warning@+1 {{Value assigned to field 'i' in implicit constructor is uninitialized}}
struct S {
  int i;
};

// The piece for the implicit copy constructor carries a reversed range.
// The old guard only compared line numbers, so a same-line reversal reached
// html::HighlightRange, whose scan then ran off the end of the buffer and
// crashed.
void reversed_range() {
  S arr[1];

  auto [a] = arr; // no-crash
  // expected-warning@-1 {{Value stored to '[a]' during its initialization is never read}}
}

// The end token's length used to be added twice, so the highlight reached over the ';'.
// CHECK-DAG: <span class="mrange">&amp;<span class='string_literal'>"abc"</span></span>;
void overshoot() {
  const char (*q)[4];
  q = &"abc"; // expected-warning {{Value stored to 'q' is never read}}
}

// A range ending inside a macro expansion now covers the whole macro use, so
// the highlight nests correctly around the 'macro' element instead of ending
// inside it.
// CHECK-DAG: <span class="mrange">{{.*}}<span class='macro'>DEREF(p)<span class='macro_popup'>(*(p))</span></span>{{.*}}</span>
#define DEREF(p) (*(p))

void ends_inside_expansion(int *p) {
  if (!p)
    DEREF(p) = 1; // expected-warning {{Dereference of null pointer}}
}

