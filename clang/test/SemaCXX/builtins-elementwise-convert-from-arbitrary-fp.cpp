// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 \
// RUN:   -fsyntax-only -verify %s

template <typename Src> float convert(Src src) {
  return __builtin_elementwise_convert_from_f8e5m2_f32(src); // expected-error {{argument type 'unsigned short' must be exactly 8 bits wide to hold an 'f8e5m2' encoding}}
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
