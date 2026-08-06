// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fsyntax-only \
// RUN:   -verify=expected,aarch64 %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -verify=expected,powerpc %s

typedef unsigned char v4u8 __attribute__((ext_vector_type(4)));
typedef unsigned short v4u16 __attribute__((ext_vector_type(4)));
typedef float v4f32 __attribute__((ext_vector_type(4)));
typedef float v2f32 __attribute__((ext_vector_type(2)));

const char *runtime_format;

void test_format(unsigned char b) {
  (void)__builtin_convert_from_arbitrary_fp(b, runtime_format, float); // expected-error {{expression is not a string literal}}
  (void)__builtin_convert_from_arbitrary_fp(b, "", float);             // expected-error {{'' is not a supported arbitrary floating-point format}}
  (void)__builtin_convert_from_arbitrary_fp(b, "float8e5m2", float);   // expected-error {{'float8e5m2' is not a supported arbitrary floating-point format}}
  (void)__builtin_convert_from_arbitrary_fp(b, u8"Float8E5M2", float); // expected-error {{format argument to __builtin_convert_from_arbitrary_fp must be an ordinary string literal}}
}

void test_arity(unsigned char b) {
  (void)__builtin_convert_from_arbitrary_fp(b, "Float8E5M2"); // expected-error {{expected ','}}
  (void)__builtin_convert_from_arbitrary_fp(b, float);        // expected-error {{expected expression}}
}

void test_width(unsigned short s, unsigned char b, unsigned _BitInt(4) b4) {
  (void)__builtin_convert_from_arbitrary_fp(s, "Float8E5M2", float);   // expected-error {{argument type 'unsigned short' must be an integer type 8 bits wide to match format 'Float8E5M2'}}
  (void)__builtin_convert_from_arbitrary_fp(b, "Float6E3M2FN", float); // expected-error {{argument type 'unsigned char' must be an integer type 6 bits wide to match format 'Float6E3M2FN'}}
  (void)__builtin_convert_from_arbitrary_fp(b4, "Float8E5M2", float);  // expected-error {{argument type 'unsigned _BitInt(4)' must be an integer type 8 bits wide to match format 'Float8E5M2'}}
}

void test_operand_types(unsigned char b, float f, void *p) {
  (void)__builtin_convert_from_arbitrary_fp(f, "Float8E5M2", float); // expected-error {{first argument to __builtin_convert_from_arbitrary_fp must be an integer type or a vector of integer types}}
  (void)__builtin_convert_from_arbitrary_fp(p, "Float8E5M2", float); // expected-error {{first argument to __builtin_convert_from_arbitrary_fp must be an integer type or a vector of integer types}}
  (void)__builtin_convert_from_arbitrary_fp(b, "Float8E5M2", int);   // expected-error {{third argument to __builtin_convert_from_arbitrary_fp must be a floating-point type or a vector of floating-point types}}
  (void)__builtin_convert_from_arbitrary_fp(b, "Float8E5M2", void);  // expected-error {{third argument to __builtin_convert_from_arbitrary_fp must be a floating-point type or a vector of floating-point types}}
}

void test_unsupported_destinations(unsigned char b) {
#ifdef __x86_64__
  (void)__builtin_convert_from_arbitrary_fp(b, "Float8E5M2", __float128); // expected-error {{destination type '__float128' is not supported by __builtin_convert_from_arbitrary_fp}}
  (void)__builtin_convert_from_arbitrary_fp(b, "Float8E5M2", long double); // expected-error {{destination type 'long double' is not supported by __builtin_convert_from_arbitrary_fp}}
#endif
#ifdef __aarch64__
  (void)__builtin_convert_from_arbitrary_fp(b, "Float8E5M2", long double); // aarch64-error {{destination type 'long double' is not supported by __builtin_convert_from_arbitrary_fp}}
  (void)__builtin_convert_from_arbitrary_fp(b, "Float8E5M2", __mfp8);      // aarch64-error {{destination type '__mfp8' is not supported by __builtin_convert_from_arbitrary_fp}}
#endif
#ifdef __powerpc__
  (void)__builtin_convert_from_arbitrary_fp(b, "Float8E5M2", long double); // powerpc-error {{destination type 'long double' is not supported by __builtin_convert_from_arbitrary_fp}}
  (void)__builtin_convert_from_arbitrary_fp(b, "Float8E5M2", __ibm128);    // powerpc-error {{destination type '__ibm128' is not supported by __builtin_convert_from_arbitrary_fp}}
#endif
}

void test_vectors(unsigned char b, v4u8 vb) {
  (void)__builtin_convert_from_arbitrary_fp(vb, "Float8E5M2", float); // expected-error {{third argument to __builtin_convert_from_arbitrary_fp must be of vector type}}
  (void)__builtin_convert_from_arbitrary_fp(b, "Float8E5M2", v4f32);  // expected-error {{first argument to __builtin_convert_from_arbitrary_fp must be of vector type}}
  (void)__builtin_convert_from_arbitrary_fp(vb, "Float8E5M2", v2f32); // expected-error {{floating-point and integer arguments to __builtin_convert_from_arbitrary_fp must have the same number of elements}}
}

void test_vector_width(v4u16 v) {
  (void)__builtin_convert_from_arbitrary_fp(v, "Float8E5M2", v4f32); // expected-error {{vector element type 'unsigned short' must be 8 bits wide to match format 'Float8E5M2'}}
}

void test_volatile_source(volatile unsigned char *b) {
  __builtin_assume(
      __builtin_convert_from_arbitrary_fp(*b, "Float8E5M2", float)); // expected-warning {{assumption is ignored because it contains (potential) side-effects}}
}

// Formats that are valid but that no target lowers yet are accepted here; the
// backend reports them.
void test_accepted(unsigned char b, unsigned _BitInt(6) b6, unsigned _BitInt(4) b4,
                   v4u8 vb) {
  (void)__builtin_convert_from_arbitrary_fp(b, "Float8E5M2", float);
  (void)__builtin_convert_from_arbitrary_fp(b, "Float8E5M2FNUZ", float);
  (void)__builtin_convert_from_arbitrary_fp(b, "Float8E4M3", float);
  (void)__builtin_convert_from_arbitrary_fp(b, "Float8E4M3FN", float);
  (void)__builtin_convert_from_arbitrary_fp(b, "Float8E4M3FNUZ", float);
  (void)__builtin_convert_from_arbitrary_fp(b, "Float8E4M3B11FNUZ", float);
  (void)__builtin_convert_from_arbitrary_fp(b, "Float8E3M4", float);
  (void)__builtin_convert_from_arbitrary_fp(b, "Float8E8M0FNU", float);
  (void)__builtin_convert_from_arbitrary_fp(b6, "Float6E3M2FN", float);
  (void)__builtin_convert_from_arbitrary_fp(b6, "Float6E2M3FN", float);
  (void)__builtin_convert_from_arbitrary_fp(b4, "Float4E2M1FN", float);
  (void)__builtin_convert_from_arbitrary_fp((signed char)b, "Float8E5M2", float);
  (void)__builtin_convert_from_arbitrary_fp(b, ("Float8E5M2"), double);
  (void)__builtin_convert_from_arbitrary_fp(vb, "Float8E5M2", v4f32);
}

_Static_assert(__has_builtin(__builtin_convert_from_arbitrary_fp), "");
