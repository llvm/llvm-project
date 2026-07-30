// RUN: %clang_analyze_cc1 -x c++ -analyzer-checker=debug.ExprInspection \
// RUN:    -analyzer-config support-symbolic-integer-casts=false \
// RUN:    -verify %s

// RUN: %clang_analyze_cc1 -x c++ -analyzer-checker=debug.ExprInspection \
// RUN:    -analyzer-config support-symbolic-integer-casts=true \
// RUN:    -verify %s

template <typename T>
void clang_analyzer_dump(T);

void test_no_redundant_floating_point_cast(int n) {

  double D = n / 30;
  clang_analyzer_dump(D); // expected-warning{{(double) ((reg_$0<int n>) / 30)}}

  // There are two cast operations evaluated above:
  // 1. (n / 30) is cast to a double during the store of `D`.
  // 2. Then in the next line, in RegionStore::getBinding during the load of `D`.
  //
  // We should not see in the dump of the SVal any redundant casts like
  // (double) ((double) $n / 30)

}
