// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify %s

template <typename Dst, typename Src> Dst conv(Src b) {
  return __builtin_convert_from_arbitrary_fp(b, "Float8E5M2", Dst); // expected-error {{argument type 'unsigned short' must be an integer type 8 bits wide to match format 'Float8E5M2'}} \
                                                                   // expected-error {{third argument to __builtin_convert_from_arbitrary_fp must be a floating-point type or a vector of floating-point types}}
}

float instantiate_ok(unsigned char b) { return conv<float, unsigned char>(b); }
double instantiate_ok2(signed char b) { return conv<double, signed char>(b); }

// expected-note@+1 {{in instantiation of function template specialization 'conv<float, unsigned short>' requested here}}
float instantiate_bad_src(unsigned short b) { return conv<float, unsigned short>(b); }

// expected-note@+1 {{in instantiation of function template specialization 'conv<int, unsigned char>' requested here}}
int instantiate_bad_dst(unsigned char b) { return conv<int, unsigned char>(b); }

template <typename Dst> Dst non_dependent_bad_src_width(unsigned short b) {
  return __builtin_convert_from_arbitrary_fp(b, "Float8E5M2", Dst); // expected-error {{argument type 'unsigned short' must be an integer type 8 bits wide to match format 'Float8E5M2'}}
}
float instantiate_non_dependent_bad_src_width(unsigned short b) {
  return non_dependent_bad_src_width<float>(b);
}

template <typename Dst> Dst non_dependent_bad_src_type(float b) {
  return __builtin_convert_from_arbitrary_fp(b, "Float8E5M2", Dst); // expected-error {{first argument to __builtin_convert_from_arbitrary_fp must be an integer type or a vector of integer types}}
}
float instantiate_non_dependent_bad_src_type(float b) {
  return non_dependent_bad_src_type<float>(b);
}

template <typename Src> int non_dependent_bad_dst(Src b) {
  return __builtin_convert_from_arbitrary_fp(b, "Float8E5M2", int); // expected-error {{third argument to __builtin_convert_from_arbitrary_fp must be a floating-point type or a vector of floating-point types}}
}
int instantiate_non_dependent_bad_dst(unsigned char b) {
  return non_dependent_bad_dst(b);
}

// The format string is validated before instantiation.
template <typename Dst> Dst bad_format(unsigned char b) {
  return __builtin_convert_from_arbitrary_fp(b, "Nope", Dst); // expected-error {{'Nope' is not a supported arbitrary floating-point format}}
}

struct S {
  static constexpr const char *fmt = "Float8E5M2";
};

void non_literal_format(unsigned char b) {
  (void)__builtin_convert_from_arbitrary_fp(b, S::fmt, float); // expected-error {{expression is not a string literal}}
}

// A dependent source operand does not confuse the format check.
template <typename Src> float dependent_src(Src b) {
  return __builtin_convert_from_arbitrary_fp(b, "Float8E4M3FN", float);
}
float use_dependent_src(unsigned char b) { return dependent_src(b); }

void noexcept_check(unsigned char b) {
  static_assert(
      noexcept(__builtin_convert_from_arbitrary_fp(b, "Float8E5M2", float)), "");
}
