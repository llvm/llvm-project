// RUN: %clang_cc1 -verify -pedantic -fsyntax-only -cl-std=CL1.2 %s
// RUN: %clang_cc1 -verify -pedantic -fsyntax-only -cl-std=CL2.0 %s

typedef image1d_t img1d_ro_default; // expected-note {{previously declared 'read_only' here}}

typedef write_only image1d_t img1d_wo; // expected-note {{previously declared 'write_only' here}}
typedef read_only image1d_t img1d_ro;

#if __OPENCL_C_VERSION__ >= 200
typedef read_write image1d_t img1d_rw;
#endif

typedef int Int;
typedef read_only int IntRO; // expected-error {{access qualifier can only be used for pipe and image type}}


void myWrite(write_only image1d_t); // expected-note {{passing argument to parameter here}} expected-note {{passing argument to parameter here}}
void myRead(read_only image1d_t); // expected-note {{passing argument to parameter here}}

#if __OPENCL_C_VERSION__ >= 200
void myReadWrite(read_write image1d_t);
#else
void myReadWrite(read_write image1d_t); // expected-error {{access qualifier 'read_write' can not be used for '__read_write image1d_t' prior to OpenCL version 2.0}}
#endif


kernel void k1(img1d_wo img) {
  myRead(img); // expected-error {{passing 'img1d_wo' (aka '__write_only image1d_t') to parameter of incompatible type '__read_only image1d_t'}}
}

kernel void k2(img1d_ro img) {
  myWrite(img); // expected-error {{passing 'img1d_ro' (aka '__read_only image1d_t') to parameter of incompatible type '__write_only image1d_t'}}
}

kernel void k3(img1d_wo img) {
  myWrite(img);
}

#if __OPENCL_C_VERSION__ >= 200
kernel void k4(img1d_rw img) {
  myReadWrite(img);
}
#endif

kernel void k5(img1d_ro_default img) {
  myWrite(img); // expected-error {{passing 'img1d_ro_default' (aka '__read_only image1d_t') to parameter of incompatible type '__write_only image1d_t'}}
}

kernel void k6(img1d_ro img) {
  myRead(img);
}

kernel void k7(read_only img1d_wo img){} // expected-error {{multiple access qualifiers}}

kernel void k8(write_only img1d_ro_default img){} // expected-error {{multiple access qualifiers}}

kernel void k9(read_only int i){} // expected-error{{access qualifier can only be used for pipe and image type}}

kernel void k10(read_only Int img){} // expected-error {{access qualifier can only be used for pipe and image type}}

kernel void k11(read_only write_only image1d_t i){} // expected-error{{multiple access qualifiers}}

kernel void k12(read_only read_only image1d_t i){} // expected-error{{multiple access qualifiers}}

#if __OPENCL_C_VERSION__ >= 200
kernel void k13(read_write pipe int i){} // expected-error{{access qualifier 'read_write' can not be used for 'pipe int'}}
#else
kernel void k13(__read_write image1d_t i){} // expected-error{{access qualifier '__read_write' can not be used for '__read_write image1d_t' prior to OpenCL version 2.0}}
#endif
