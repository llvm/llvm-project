// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 \
// RUN:   -fsyntax-only -verify %s

template <typename Src> float convert(Src src) {
  return __builtin_elementwise_convert_from_f8e5m2_f32(src); // expected-error {{1st argument must be a scalar or vector of 8-bit integer types (was 'unsigned short')}}
}

float instantiate_valid(unsigned char src) { return convert(src); }

// expected-note@+1 {{in instantiation of function template specialization 'convert<unsigned short>' requested here}}
float instantiate_invalid(unsigned short src) { return convert(src); }

template <typename Src>
auto deduced_result(Src src)
    -> decltype(__builtin_elementwise_convert_from_f8e5m2_f32(src)) {
  return __builtin_elementwise_convert_from_f8e5m2_f32(src);
}

static_assert(
    __is_same(decltype(deduced_result((unsigned char)0)), float), "");

using v4u8 = unsigned char __attribute__((ext_vector_type(4)));
using v4f32 = float __attribute__((ext_vector_type(4)));
static_assert(__is_same(decltype(deduced_result(v4u8{})), v4f32), "");

template <int N> float convert_constant() {
  return __builtin_elementwise_convert_from_f8e5m2_f32(N); // expected-error {{1st argument must be a scalar or vector of 8-bit integer types (was 'int')}}
}

float instantiate_constant() { return convert_constant<0x38>(); }

// expected-note@+1 {{in instantiation of function template specialization 'convert_constant<256>' requested here}}
float instantiate_constant_invalid() { return convert_constant<0x100>(); }

namespace std {
enum class byte : unsigned char {};
} // namespace std

enum class other_byte : unsigned char {};

void test_byte(std::byte b, other_byte o) {
  static_assert(
      __is_same(decltype(__builtin_elementwise_convert_from_f8e5m2_f32(b)),
                float),
      "");
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(o); // expected-error {{1st argument must be a scalar or vector of 8-bit integer types (was 'other_byte')}}
}

void noexcept_check(unsigned char src) {
  static_assert(
      noexcept(__builtin_elementwise_convert_from_f8e5m2_f32(src)), "");
}

static_assert(
    !__has_constexpr_builtin(
        __builtin_elementwise_convert_from_f8e5m2_f32), "");

constexpr float constant_evaluation_is_deferred =
    __builtin_elementwise_convert_from_f8e5m2_f32(
        (unsigned char)0); // expected-error@-1 {{constexpr variable 'constant_evaluation_is_deferred' must be initialized by a constant expression}}
