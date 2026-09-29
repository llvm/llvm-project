// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -DAST_DUMP -ast-dump -ast-dump-filter call_expr %s | FileCheck %s --check-prefix=AST

#ifdef AST_DUMP

// The builtin remains an ordinary CallExpr. Its operand undergoes an
// lvalue-to-rvalue conversion without an integer promotion.
float call_expr(unsigned char bits) {
  return __builtin_elementwise_convert_from_f8e5m2_f32(bits);
}

// AST-LABEL: FunctionDecl {{.*}} call_expr 'float (unsigned char)'
// AST: ReturnStmt
// AST-NEXT: CallExpr {{.*}} 'float'
// AST: ImplicitCastExpr {{.*}} 'unsigned char' <LValueToRValue>
// AST-NEXT: DeclRefExpr {{.*}} 'unsigned char' lvalue ParmVar {{.*}} 'bits' 'unsigned char'

#else

using uchar = unsigned char;
using uchar4 = uchar __attribute__((ext_vector_type(4)));
using half4 = _Float16 __attribute__((ext_vector_type(4)));
using bfloat4 = __bf16 __attribute__((ext_vector_type(4)));
using float4 = float __attribute__((ext_vector_type(4)));
using gnu_uchar4 = uchar __attribute__((vector_size(4)));
using gnu_half4 = _Float16 __attribute__((vector_size(8)));
using gnu_bfloat4 = __bf16 __attribute__((vector_size(8)));
using gnu_float4 = float __attribute__((vector_size(16)));

#define CHECK_TYPE(EXPR, TYPE) static_assert(__is_same(decltype(EXPR), TYPE))

void cv_qualified(const uchar &c, volatile uchar &v, const volatile uchar &cv,
                  const uchar4 &vc, volatile uchar4 &vv) {
  CHECK_TYPE(__builtin_elementwise_convert_from_f8e5m2_f16(c), _Float16);
  CHECK_TYPE(__builtin_elementwise_convert_from_f8e4m3fn_bf16(v), __bf16);
  CHECK_TYPE(__builtin_elementwise_convert_from_f8e5m3fnu_f32(cv), float);
  CHECK_TYPE(__builtin_elementwise_convert_from_f8e5m2_f16(vc), half4);
  CHECK_TYPE(__builtin_elementwise_convert_from_f8e4m3fn_bf16(vv), bfloat4);
}

template <class T> auto convert_half(T bits) {
  return __builtin_elementwise_convert_from_f8e5m2_f16(bits);
}

template <class T> auto convert_bfloat(T bits) {
  return __builtin_elementwise_convert_from_f8e4m3fn_bf16(bits);
}

template <class T> auto convert_float(T bits) {
  return __builtin_elementwise_convert_from_f8e5m3fnu_f32(bits);
}

void dependent_result_types(uchar scalar, uchar4 ext, gnu_uchar4 generic) {
  CHECK_TYPE(convert_half(scalar), _Float16);
  CHECK_TYPE(convert_bfloat(scalar), __bf16);
  CHECK_TYPE(convert_float(scalar), float);
  CHECK_TYPE(convert_half(ext), half4);
  CHECK_TYPE(convert_bfloat(ext), bfloat4);
  CHECK_TYPE(convert_float(ext), float4);
  CHECK_TYPE(convert_half(generic), gnu_half4);
  CHECK_TYPE(convert_bfloat(generic), gnu_bfloat4);
  CHECK_TYPE(convert_float(generic), gnu_float4);
}

// Value dependence does not change the argument's known 8-bit integer type.
template <uchar Bits> auto convert_value() {
  return __builtin_elementwise_convert_from_f8e4m3fn_f32(Bits);
}
CHECK_TYPE(convert_value<0x38>(), float);

enum byte_enum : unsigned char { byte_zero };
enum class scoped_byte_enum : unsigned char { zero };
struct convertible {
  operator unsigned char() const;
};

void invalid_types(bool b, byte_enum e, scoped_byte_enum se, convertible c) {
  __builtin_elementwise_convert_from_f8e5m2_f32(b);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m2_f32(e);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m2_f32(se);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m2_f32(c);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
}

template <class T> auto invalid_dependent(T bits) {
  return __builtin_elementwise_convert_from_f8e5m2_f32(bits);
  // expected-error@-1 3 {{must be an 8-bit integer or a vector of 8-bit integers}}
}

template <class... T> auto dependent_arity(T... bits) {
  return __builtin_elementwise_convert_from_f8e4m3fn_f32(bits...);
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
  // expected-error@-2 {{too many arguments to function call, expected 1, have 2}}
}

void instantiate_errors() {
  invalid_dependent(0);
  // expected-note@-1 {{in instantiation of function template specialization 'invalid_dependent<int>' requested here}}
  invalid_dependent(false);
  // expected-note@-1 {{in instantiation of function template specialization 'invalid_dependent<bool>' requested here}}
  invalid_dependent(byte_zero);
  // expected-note@-1 {{in instantiation of function template specialization 'invalid_dependent<byte_enum>' requested here}}
  dependent_arity();
  // expected-note@-1 {{in instantiation of function template specialization 'dependent_arity<>' requested here}}
  dependent_arity(uchar{}, uchar{});
  // expected-note@-1 {{in instantiation of function template specialization 'dependent_arity<unsigned char, unsigned char>' requested here}}
}

constexpr _Float16 constant_half = __builtin_elementwise_convert_from_f8e5m2_f16(uchar{});
// expected-error@-1 {{constexpr variable 'constant_half' must be initialized by a constant expression}}
constexpr __bf16 constant_bfloat = __builtin_elementwise_convert_from_f8e4m3fn_bf16(uchar{});
// expected-error@-1 {{constexpr variable 'constant_bfloat' must be initialized by a constant expression}}
constexpr float constant_float = __builtin_elementwise_convert_from_f8e5m3fnu_f32(uchar{});
// expected-error@-1 {{constexpr variable 'constant_float' must be initialized by a constant expression}}

static_assert(__builtin_elementwise_convert_from_f8e5m2_f32(uchar{}) == 0.0f);
// expected-error@-1 {{static assertion expression is not an integral constant expression}}

#endif
