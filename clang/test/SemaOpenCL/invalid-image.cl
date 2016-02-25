// RUN: %clang_cc1 -verify %s

void test1(image1d_t *i){} // expected-error {{pointer to type 'image1d_t' is invalid in OpenCL}}

void test2(image1d_t i) {
  image1d_t ti; // expected-error {{type 'image1d_t' can only be used as a function parameter}}
  image1d_t ai[] = {i,i};// expected-error {{array of 'image1d_t' type is invalid in OpenCL}}
}
