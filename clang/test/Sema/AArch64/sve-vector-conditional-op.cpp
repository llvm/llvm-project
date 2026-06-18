// RUN: %clang_cc1 %s -fsyntax-only -triple aarch64-none-linux-gnu -target-feature +sve -verify

typedef int fixed_vector __attribute__((vector_size(4)));

auto error_fixed_vector_result(__SVBool_t svbool, fixed_vector a, fixed_vector b) {
  // expected-error@+1 {{vector condition type '__SVBool_t' and result type 'fixed_vector' (vector of 1 'int' value) do not have the same number of elements}}
  return svbool ? a : b;
}

auto error_void_result(__SVBool_t svbool) {
  // expected-error@+1 {{GNU vector conditional operand cannot be void}}
  return svbool ? (void)0 : (void)1;
}

auto error_sve_splat_result_unsupported(__SVBool_t svbool, long long a, long long b) {
  // expected-error@+1 {{scalar type 'long long' not supported with vector condition type '__SVBool_t'}}
  return svbool ? a : b;
}

auto error_sve_vector_result_matched_element_count(__SVBool_t svbool, __SVUint32_t a, __SVUint32_t b) {
  // expected-error@+1 {{vector condition type '__SVBool_t' and result type '__SVUint32_t' do not have the same number of elements}}
  return svbool ? a : b;
}

auto error_fixed_cond_mixed_scalar_and_vector_operands(fixed_vector cond, unsigned char a, __SVUint8_t b) {
  // expected-error@+1 {{cannot mix vectors and sizeless vectors in a vector conditional}}
  return cond ? a : b;
}

auto error_scalable_cond_mixed_scalar_and_vector_operands(__SVBool_t svbool, unsigned char a, fixed_vector b) {
  // expected-error@+1 {{cannot mix vectors and sizeless vectors in a vector conditional}}
  return svbool ? a : b;
}

// The following cases should be supported:

__SVBool_t cond_svbool(__SVBool_t a, __SVBool_t b) {
    return a < b ? a : b;
}

__SVFloat32_t cond_svf32(__SVFloat32_t a, __SVFloat32_t b) {
    return a < b ? a : b;
}

__SVUint64_t cond_u64_splat(__SVUint64_t a) {
    return a < 1ul ? a : 1ul;
}
