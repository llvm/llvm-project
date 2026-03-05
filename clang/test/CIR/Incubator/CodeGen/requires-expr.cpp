// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Test RequiresExpr as a boolean expression
bool test_requires_simple() {
  // CHECK-LABEL: cir.func{{.*}} @{{.*}}test_requires_simplev
  // CHECK: %{{.*}} = cir.const #true
  return requires { 1 + 1; };
}

template <typename T>
bool test_requires_param() {
  return requires(T t) { t + 1; };
}

bool use_requires_param() {
  // CHECK-LABEL: cir.func{{.*}} @{{.*}}use_requires_paramv
  // Instantiation with int should succeed
  return test_requires_param<int>();
  // CHECK: cir.call @{{.*}}test_requires_paramIiEbv
}

// Test requires expression with multiple requirements
bool test_requires_multiple() {
  // CHECK-LABEL: cir.func{{.*}} @{{.*}}test_requires_multiplev
  // CHECK: %{{.*}} = cir.const #true
  return requires {
    1 + 1;
    2 * 2;
  };
}

// Test requires expression in if statement
int test_requires_in_if() {
  // CHECK-LABEL: cir.func{{.*}} @{{.*}}test_requires_in_ifv
  if (requires { 1 + 1; }) {
    // CHECK: %{{.*}} = cir.const #true
    // CHECK: cir.if %{{.*}} {
    return 1;
  }
  return 0;
}

// Test requires expression that should fail
template <typename T>
bool test_requires_fail() {
  return requires { T::nonexistent_member; };
}

bool use_requires_fail() {
  // CHECK-LABEL: cir.func{{.*}} @{{.*}}use_requires_failv
  // Should return false for int (no member named nonexistent_member)
  return test_requires_fail<int>();
  // CHECK: cir.call @{{.*}}test_requires_failIiEbv
}

// Test nested requires
bool test_nested_requires() {
  // CHECK-LABEL: cir.func{{.*}} @{{.*}}test_nested_requiresv
  // CHECK: %{{.*}} = cir.const #true
  return requires {
    requires true;
  };
}

// Use in constexpr context
constexpr bool can_add_int = requires(int a, int b) { a + b; };

int use_constexpr_requires() {
  // CHECK-LABEL: cir.func{{.*}} @{{.*}}use_constexpr_requiresv
  if (can_add_int) {
    return 42;
  }
  return 0;
}
