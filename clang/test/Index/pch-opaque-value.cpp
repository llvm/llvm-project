struct S {
  S();
  S(int);
  S(const S &);
  ~S();
};

void f() {
  try {
  } catch (S e) {
  }
}

// RUN: c-index-test -write-pch %t.pch %s
// RUN: c-index-test -test-load-tu-usrs %t.pch local | FileCheck %s
// CHECK: pch-opaque-value.cpp c:pch-opaque-value.cpp@86@F@f#@e Extent=[10:12 - 10:15]
