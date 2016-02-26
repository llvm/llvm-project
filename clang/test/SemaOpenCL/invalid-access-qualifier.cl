// RUN: %clang_cc1 -verify %s
// RUN: %clang_cc1 -verify -cl-std=CL2.0 -DCL20 %s

void test1(read_only int i){} // expected-error{{access qualifier can only be used for pipe and image type}}

void test2(read_only write_only image1d_t i){} // expected-error{{multiple access qualifiers}}

void test3(read_only read_only image1d_t i){} // expected-error{{multiple access qualifiers}}

#ifdef CL20
void test4(read_write pipe int i){} // expected-error{{access qualifier 'read_write' can not be used for 'pipe'}}
#else
void test4(__read_write image1d_t i){} // expected-error{{access qualifier '__read_write' can not be used for 'image1d_t' earlier than OpenCL2.0 version}}
#endif
