// RUN: %clang_cc1 -verify %s

void test1(image1d_t *i) {} // expected-error{{pointer to type '__read_only image1d_t' is invalid in OpenCL}}

void test2(image1d_t i) {
  image1d_t ti;            // expected-error{{type '__read_only image1d_t' can only be used as a function parameter}}
  image1d_t ai[] = {i, i}; // expected-error{{array of '__read_only image1d_t' type is invalid in OpenCL}}
  i=i; // expected-error{{invalid operands to binary expression ('__read_only image1d_t' and '__read_only image1d_t')}}
  i+1; // expected-error{{invalid operands to binary expression ('__read_only image1d_t' and 'int')}}
  &i; // expected-error{{invalid argument type '__read_only image1d_t' to unary expression}}
  *i; // expected-error{{invalid argument type '__read_only image1d_t' to unary expression}}
}

image1d_t test3() {} // expected-error{{declaring function return value of type '__read_only image1d_t' is not allowed}}
