// RUN: %clang++ -std=gnu++11 -O2 -ffast-math -g %s -o %t
// RUN: %dexter -w --use-script \
// RUN:     --binary %t %dexter_lldb_args -- %s | FileCheck %s
// RUN: %clang++ -std=gnu++11 -O0 -ffast-math -g %s -o %t
// RUN: %dexter -w --use-script \
// RUN:     --binary %t %dexter_lldb_args -- %s | FileCheck %s

// REQUIRES: lldb
// Currently getting intermittent failures on darwin.
// UNSUPPORTED: system-windows, system-darwin

//// Check that the debugging experience with __attribute__((optnone)) at O2
//// matches O0. Test scalar floating point arithmetic with -ffast-math.

//// Example of strength reduction.
//// The division by 10.0f can be rewritten as a multiply by 0.1f.
//// A / 10.f ==> A * 0.1f
//// This is safe with fastmath since we treat the two operations
//// as equally precise. However we don't want this to happen
//// with optnone.
__attribute__((optnone))
float test_fdiv(float A) {
  float result;
  result = A / 10.f; // !dex_label fdiv_assign
  return result;     // !dex_label fdiv_ret
}

//// (A * B) - (A * C) ==> A * (B - C)
__attribute__((optnone))
float test_distributivity(float A, float B, float C) {
  float result;
  float op1 = A * B;
  float op2 = A * C;  // !dex_label distributivity_op2
  result = op1 - op2; // !dex_label distributivity_result
  return result;      // !dex_label distributivity_ret
}

//// (A + B) + C  == A + (B + C)
//// therefore, ((A + B) + C) + (A + (B + C)))
//// can be rewritten as
//// 2.0f * ((A + B) + C)
//// Clang is currently unable to spot this optimization
//// opportunity with fastmath.
__attribute__((optnone))
float test_associativity(float A, float B, float C) {
  float result;
  float op1 = A + B;
  float op2 = B + C;
  op1 += C;           // !dex_label associativity_op1
  op2 += A;
  result = op1 + op2; // !dex_label associativity_result
  return result;      // !dex_label associativity_ret
}

//// With fastmath, the ordering of instructions doesn't matter
//// since we work under the assumption that there is no loss
//// in precision. This simplifies things for the optimizer which
//// can then decide to reorder instructions and fold
//// redundant operations like this:
////   A += 5.0f
////   A -= 5.0f
////    -->
////   A
//// This function can be simplified to a return A + B.
__attribute__((optnone))
float test_simplify_fp_operations(float A, float B) {
  float result = A + 10.0f; // !dex_label fp_operations_result
  result += B;              // !dex_label fp_operations_add
  result -= 10.0f;
  return result;            // !dex_label fp_operations_ret
}

//// Again, this is a simple return A + B.
//// Clang is unable to spot the opportunity to fold the code sequence.
__attribute__((optnone))
float test_simplify_fp_operations_2(float A, float B, float C) {
  float result = A + C; // !dex_label fp_operations_2_result
  result += B;
  result -= C;          // !dex_label fp_operations_2_subtract
  return result;        // !dex_label fp_operations_2_ret
}

int main() {
  float result = test_fdiv(4.0f);
  result += test_distributivity(4.0f, 5.0f, 6.0f);
  result += test_associativity(4.0f, 5.0f, 6.0f);
  result += test_simplify_fp_operations(8.25, result);
  result += test_simplify_fp_operations_2(9.12, result, 1002.111);
  return static_cast<int>(result);
}

// CHECK-DAG: seen_values: 20
// CHECK-DAG: correct_step_coverage: 100.0%

/*
---
!where {function: test_fdiv}:
  !and {lines: !label fdiv_assign}:
    !value A: 4
  !and {lines: !label fdiv_ret}:
    !value result: "0.400000006"
!where {function: test_distributivity}:
  !and {lines: !label distributivity_op2}:
    !value op1: 20
  !and {lines: !label distributivity_result}:
    !value op2: 24
  !and {lines: !label distributivity_ret}:
    !value result: -4
!where {function: test_associativity}:
  !and {lines: !range [!label associativity_op1, !label associativity_result]}:
    !value op1: [9, 15]
    !value op2: [11, 15]
  !and {lines: !label associativity_ret}:
    !value result: 30
!where {function: test_simplify_fp_operations}:
  !and {lines: !label fp_operations_result}:
    !value A: "8.25"
    !value B: "26.3999996"
  !and {lines: !range [!label fp_operations_add, !label fp_operations_ret]}:
    !value result: ["18.25", "44.6500015", "34.6500015"]
!where {function: test_simplify_fp_operations_2}:
  !and {lines: !label fp_operations_2_result}:
    !value A: '9.11999988'
    !value B: '61.050003'
    !value C: '1002.11102'
  ? !and
    lines: !range [!label fp_operations_2_subtract, !label fp_operations_2_ret]
  : !value result: ["1072.28101", "70.1699829"]
...
*/
