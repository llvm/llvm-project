// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-argument-comment %t

struct Base {
  explicit Base(int val) {}
};

struct Over : public Base {
  using Base::Base;
};

int wrong() {
  Base b{/*vall=*/2};
// CHECK-NOTES: [[@LINE-1]]:10: warning: argument name 'vall' in comment does not match parameter name 'val'
// CHECK-NOTES: [[@LINE-10]]:21: note: 'val' declared here
// CHECK-FIXES: Base b{/*val=*/2};

  Over o{/*vall=*/3};
// CHECK-NOTES: [[@LINE-1]]:10: warning: argument name 'vall' in comment does not match parameter name 'val'
// CHECK-NOTES: [[@LINE-15]]:21: note: 'val' declared here
// CHECK-NOTES: [[@LINE-12]]:15: note: actual callee ('Base') is declared here
// CHECK-FIXES: Over o{/*val=*/3};
}
