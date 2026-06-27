// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-argument-comment %t

struct Base {
  explicit Base(int val) {}
};

struct Over : public Base {
  using Base::Base;
};

struct Derived : public Over {
  using Over::Over;
};

int wrong() {
  Base b{/*val2=*/2};
// CHECK-NOTES: [[@LINE-1]]:10: warning: argument name 'val2' in comment does not match parameter name 'val'
// CHECK-NOTES: [[@LINE-14]]:21: note: 'val' declared here
// CHECK-FIXES: Base b{/*val=*/2};

  Over o{/*val3=*/3};
// CHECK-NOTES: [[@LINE-1]]:10: warning: argument name 'val3' in comment does not match parameter name 'val'
// CHECK-NOTES: [[@LINE-19]]:21: note: 'val' declared here
// CHECK-NOTES: [[@LINE-16]]:15: note: actual callee ('Base') is declared here
// CHECK-FIXES: Over o{/*val=*/3};

  Derived d{/*val4=*/4};
// CHECK-NOTES: [[@LINE-1]]:13: warning: argument name 'val4' in comment does not match parameter name 'val'
// CHECK-NOTES: [[@LINE-25]]:21: note: 'val' declared here
// CHECK-NOTES: [[@LINE-18]]:15: note: actual callee ('Base') is declared here
// CHECK-FIXES: Derived d{/*val=*/4};
}
