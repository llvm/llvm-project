// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

// Don't crash if the argument to __builtin_constant_p isn't scalar.
template <typename T>
constexpr bool is_constant(const T v) {
  return __builtin_constant_p(v);
}

template <typename T>
class numeric {
 public:
  using type = T;

  template <typename S>
  constexpr numeric(S value)
      : value_(static_cast<T>(value)) {}

 private:
  const T value_;
};

bool bcp() {
  return is_constant(numeric<int>(1));
}

// PR45535
struct with_dtor {
  ~with_dtor();
};
// CHECK: define {{.*}}bcp_stmt_expr_1
bool bcp_stmt_expr_1() {
  // CHECK-NOT: call {{.*}}with_dtorD
  return __builtin_constant_p(({with_dtor wd; 123;}));
}

int do_not_call();
// CHECK: define {{.*}}bcp_stmt_expr_2
bool bcp_stmt_expr_2(int n) {
  // CHECK-NOT: call {{.*}}do_not_call
  return __builtin_constant_p(({
    // This has a side-effect due to the VLA bound, so CodeGen should fold it
    // to false.
    typedef int arr[do_not_call()];
    n;
  }));
  // CHECK-NOT: }
  // CHECK: ret i1 false
}
