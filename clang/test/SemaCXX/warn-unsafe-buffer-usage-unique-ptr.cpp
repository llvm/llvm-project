// RUN: %clang_cc1 -Wno-unused-value -Wunsafe-buffer-usage -fsafe-buffer-usage-suggestions -std=c++20 -verify=expected %s

// This debugging facility is only available in debug builds.
//
// REQUIRES: asserts

namespace std {
inline namespace __1 {
template <class T> class unique_ptr {
public:
  T &operator[](long long i) const;
};
} // namespace __1
} // namespace std

void basic_unique_ptr() {
  std::unique_ptr<int[]> p1;
  int i = 2;

  p1[0];  // This is allowed

  p1[1];  // expected-warning{{direct access using operator[] on std::unique_ptr<T[]> is unsafe due to lack of bounds checking}}

  p1[i];  // expected-warning{{direct access using operator[] on std::unique_ptr<T[]> is unsafe due to lack of bounds checking}}

  p1[i + 5]; // expected-warning{{direct access using operator[] on std::unique_ptr<T[]> is unsafe due to lack of bounds checking}}
}

// CHECK: Root # 1
// CHECK: |- DeclRefExpr # 4
// CHECK: |-- UnaryOperator(++) # 1
// CHECK: |--- CompoundStmt # 1
// CHECK: |-- ImplicitCastExpr(LValueToRValue) # 1
// CHECK: |--- BinaryOperator(+) # 1
// CHECK: |---- ParenExpr # 1
// CHECK: |----- BinaryOperator(+) # 1
// CHECK: |------ ParenExpr # 1
// CHECK: |------- UnaryOperator(*) # 1
// CHECK: |-------- CompoundStmt # 1
// CHECK: |-- BinaryOperator(-=) # 1
// CHECK: |--- CompoundStmt # 1
// CHECK: |-- UnaryOperator(--) # 1
// CHECK: |--- CompoundStmt # 1
