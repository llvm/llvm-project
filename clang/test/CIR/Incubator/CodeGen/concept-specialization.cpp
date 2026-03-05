// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

template <typename T>
concept Integral = __is_integral(T);

template <typename T>
concept Signed = Integral<T> && __is_signed(T);

// Test ConceptSpecializationExpr as a boolean value
bool test_concept_bool() {
  // CHECK-LABEL: cir.func{{.*}} @{{.*}}test_concept_boolv
  // CHECK: %{{.*}} = cir.const #true
  return Integral<int>;
}

bool test_concept_false() {
  // CHECK-LABEL: cir.func{{.*}} @{{.*}}test_concept_falsev
  // CHECK: %{{.*}} = cir.const #false
  return Integral<float>;
}

bool test_concept_compound() {
  // CHECK-LABEL: cir.func{{.*}} @{{.*}}test_concept_compoundv
  // CHECK: %{{.*}} = cir.const #true
  return Signed<int>;
}

bool test_concept_unsigned() {
  // CHECK-LABEL: cir.func{{.*}} @{{.*}}test_concept_unsignedv
  // CHECK: %{{.*}} = cir.const #false
  return Signed<unsigned>;
}

// Test in conditional
int test_concept_in_if() {
  // CHECK-LABEL: cir.func{{.*}} @{{.*}}test_concept_in_ifv
  if (Integral<int>) {
    // CHECK: %{{.*}} = cir.const #true
    // CHECK: cir.if %{{.*}} {
    return 1;
  }
  return 0;
}

// Test constexpr variable with concept
constexpr bool is_int_integral = Integral<int>;

int use_constexpr() {
  // CHECK-LABEL: cir.func{{.*}} @{{.*}}use_constexprv
  if (is_int_integral) {
    // This should be optimized to a constant true
    return 42;
  }
  return 0;
}
