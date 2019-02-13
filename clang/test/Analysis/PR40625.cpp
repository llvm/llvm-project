// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,alpha.core.CallAndMessageUnInitRefArg  %s -verify

void f(const int *end);

void g(const int (&arrr)[10]) {
  f(arrr+sizeof(arrr)); // expected-warning{{1st function call argument is a pointer to uninitialized value}}
  // FIXME: This is a false positive that should be fixed. Until then this
  //        tests the crash fix in FindLastStoreBRVisitor (beside
  //        uninit-vals.m).
}

void h() {
  int arr[10];

  g(arr);
}
