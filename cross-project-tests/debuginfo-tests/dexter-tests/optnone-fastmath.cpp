// RUN: %clang++ -std=gnu++11 -O2 -ffast-math -g %s -o %t
// RUN: %dexter --fail-lt 1.0 -w \
// RUN:     --binary %t %dexter_lldb_args -- %s
// RUN: %clang++ -std=gnu++11 -O0 -ffast-math -g %s -o %t
// RUN: %dexter --fail-lt 1.0 -w \
// RUN:     --binary %t %dexter_lldb_args -- %s

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
  result = A / 10.f;  // DexLabel('fdiv_assign')
  return result;      // DexLabel('fdiv_ret')
}
// DexExpectWatchValue('A', 4, on_line=ref('fdiv_assign'))
// DexExpectWatchValue('result', '0.400000006', on_line=ref('fdiv_ret'))

//// (A * B) - (A * C) ==> A * (B - C)
__attribute__((optnone))
float test_distributivity(float A, float B, float C) {
  float result;
  float op1 = A * B;
  float op2 = A * C;    // DexLabel('distributivity_op2')
  result = op1 - op2;   // DexLabel('distributivity_result')
  return result;        // DexLabel('distributivity_ret')
}
// DexExpectWatchValue('op1', '20', on_line=ref('distributivity_op2'))
// DexExpectWatchValue('op2', '24', on_line=ref('distributivity_result'))
// DexExpectWatchValue('result', '-4', on_line=ref('distributivity_ret'))

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
  op1 += C;           // DexLabel('associativity_op1')
  op2 += A;
  result = op1 + op2; // DexLabel('associativity_result')
  return result;      // DexLabel('associativity_ret')
}
// DexExpectWatchValue('op1', '9', '15', from_line=ref('associativity_op1'), to_line=ref('associativity_result'))
// DexExpectWatchValue('op2', '11', '15', from_line=ref('associativity_op1'), to_line=ref('associativity_result'))
// DexExpectWatchValue('result', '30', on_line=ref('associativity_ret'))

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
  float result = A + 10.0f; // DexLabel('fp_operations_result')
  result += B;              // DexLabel('fp_operations_add')
  result -= 10.0f;
  return result;            // DexLabel('fp_operations_ret')
}
// DexExpectWatchValue('A', '8.25', on_line=ref('fp_operations_result'))
// DexExpectWatchValue('B', '26.3999996', on_line=ref('fp_operations_result'))
// DexExpectWatchValue('result', '18.25', '44.6500015', '34.6500015', from_line=ref('fp_operations_add'), to_line=ref('fp_operations_ret'))

//// Again, this is a simple return A + B.
//// Clang is unable to spot the opportunity to fold the code sequence.
__attribute__((optnone))
float test_simplify_fp_operations_2(float A, float B, float C) {
  float result = A + C; // DexLabel('fp_operations_2_result')
  result += B;
  result -= C;          // DexLabel('fp_operations_2_subtract')
  return result;        // DexLabel('fp_operations_2_ret')
}
// DexExpectWatchValue('A', '9.11999988', on_line=ref('fp_operations_2_result'))
// DexExpectWatchValue('B', '61.050003', on_line=ref('fp_operations_2_result'))
// DexExpectWatchValue('C', '1002.11102', on_line=ref('fp_operations_2_result'))
// DexExpectWatchValue('result', '1072.28101', '70.1699829', from_line=ref('fp_operations_2_subtract'), to_line=ref('fp_operations_2_ret'))

int main() {
  float result = test_fdiv(4.0f);
  result += test_distributivity(4.0f, 5.0f, 6.0f);
  result += test_associativity(4.0f, 5.0f, 6.0f);
  result += test_simplify_fp_operations(8.25, result);
  result += test_simplify_fp_operations_2(9.12, result, 1002.111);
  return static_cast<int>(result);
}
