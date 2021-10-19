// RUN: %clang_cc1 -verify -cl-std=clc++ %s
// RUN: %clang_cc1 -verify %s -cl-ext=+cl_khr_gl_msaa_sharing
// RUN: %clang_cc1 -verify %s -cl-ext=-cl_khr_gl_msaa_sharing
// RUN: %clang_cc1 -verify -D=ATTR_TEST -fms-compatibility %s

void test1(image1d_t *i) {} // expected-error-re{{pointer to type '{{__generic __read_only|__read_only}} image1d_t' is invalid in OpenCL}}

void test2(image1d_t i) {
  image1d_t ti;            // expected-error{{type '__private __read_only image1d_t' can only be used as a function parameter}}
  image1d_t ai[] = {i, i}; // expected-error{{array of '__read_only image1d_t' type is invalid in OpenCL}}
  i=i; // expected-error{{invalid operands to binary expression ('__private __read_only image1d_t' and '__private __read_only image1d_t')}}
  i+1; // expected-error{{invalid operands to binary expression ('__private __read_only image1d_t' and 'int')}}
  &i; // expected-error{{invalid argument type '__private __read_only image1d_t' to unary expression}}
  *i; // expected-error{{invalid argument type '__private __read_only image1d_t' to unary expression}}
}

image1d_t test3() {} // expected-error{{declaring function return value of type '__read_only image1d_t' is not allowed}}

#ifdef ATTR_TEST
// Test case for an infinite loop bug.
kernel void foob(read_only __ptr32  image2d_t i) { } // expected-error{{'__ptr32' attribute only applies to pointer arguments}}
#endif

typedef int image1d_t; // expected-error{{cannot combine with previous 'int' declaration specifier}} expected-warning{{typedef requires a name}}
typedef int image2d_t; // expected-error{{cannot combine with previous 'int' declaration specifier}} expected-warning{{typedef requires a name}}
typedef int image3d_t; // expected-error{{cannot combine with previous 'int' declaration specifier}} expected-warning{{typedef requires a name}}
typedef int image1d_array_t; // expected-error{{cannot combine with previous 'int' declaration specifier}} expected-warning{{typedef requires a name}}
typedef int image2d_array_t; // expected-error{{cannot combine with previous 'int' declaration specifier}} expected-warning{{typedef requires a name}}
typedef int image2d_depth_t; // expected-error{{cannot combine with previous 'int' declaration specifier}} expected-warning{{typedef requires a name}}
typedef int image1d_buffer_t; // expected-error{{cannot combine with previous 'int' declaration specifier}} expected-warning{{typedef requires a name}}

// Image types from 'cl_khr_gl_msaa_sharing' are not reserved identifiers.
typedef int image2d_msaa_t;
typedef int image2d_array_msaa_t;
typedef int image2d_msaa_depth_t;
typedef int image2d_array_msaa_depth_t;
#ifdef cl_khr_gl_msaa_sharing
// expected-error@-5{{cannot combine with previous 'int' declaration specifier}} expected-warning@-5{{typedef requires a name}}
// expected-error@-5{{cannot combine with previous 'int' declaration specifier}} expected-warning@-5{{typedef requires a name}}
// expected-error@-5{{cannot combine with previous 'int' declaration specifier}} expected-warning@-5{{typedef requires a name}}
// expected-error@-5{{cannot combine with previous 'int' declaration specifier}} expected-warning@-5{{typedef requires a name}}
#endif
void foo(image2d_msaa_t i);
