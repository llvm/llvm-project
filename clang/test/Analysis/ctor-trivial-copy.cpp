// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config c++-inlining=constructors -verify %s


template<typename T>
void clang_analyzer_dump(T&);

struct aggr {
  int x;
  int y;
};

struct empty {
};

void test_copy_return() {
  aggr s1 = {1, 2};
  aggr const& cr1 = aggr(s1);
  clang_analyzer_dump(cr1); // expected-warning-re {{&lifetime_extended_object{aggr, cr1, S{{[0-9]+}}} }}

  empty s2;
  empty const& cr2 = empty{s2};
  clang_analyzer_dump(cr2); // expected-warning-re {{&lifetime_extended_object{empty, cr2, S{{[0-9]+}}} }}
}

void test_assign_return() {
  aggr s1 = {1, 2};
  aggr d1;
  clang_analyzer_dump(d1 = s1); // expected-warning {{&d1 }}

  empty s2;
  empty d2;
  clang_analyzer_dump(d2 = s2); // expected-warning {{&d2 }} was Unknown
}

