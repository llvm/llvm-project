// RUN: %clang_cc1 -verify -pedantic -fsyntax-only -cl-std=CL1.2 %s -cl-ext=-cl_khr_3d_image_writes
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
  myRead(img); // expected-error {{passing '__private img1d_wo' (aka '__private __write_only image1d_t') to parameter of incompatible type '__read_only image1d_t'}}
}

kernel void k2(img1d_ro img) {
  myWrite(img); // expected-error {{passing '__private img1d_ro' (aka '__private __read_only image1d_t') to parameter of incompatible type '__write_only image1d_t'}}
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
  myWrite(img); // expected-error {{passing '__private img1d_ro_default' (aka '__private __read_only image1d_t') to parameter of incompatible type '__write_only image1d_t'}}
}

kernel void k6(img1d_ro img) {
  myRead(img);
}

kernel void k7(read_only img1d_wo img){} // expected-error {{multiple access qualifiers}}

kernel void k8(write_only img1d_ro_default img){} // expected-error {{multiple access qualifiers}}

kernel void k9(read_only int i){} // expected-error{{access qualifier can only be used for pipe and image type}}

kernel void k10(read_only Int img){} // expected-error {{access qualifier can only be used for pipe and image type}}

kernel void k11(read_only write_only image1d_t i){} // expected-error{{multiple access qualifiers}}

kernel void k12(read_only read_only image1d_t i){} // expected-warning {{duplicate 'read_only' declaration specifier}}

#if __OPENCL_C_VERSION__ >= 200
kernel void k13(read_write pipe int i){} // expected-error{{access qualifier 'read_write' can not be used for 'read_only pipe int'}}
#else
kernel void k13(__read_write image1d_t i){} // expected-error{{access qualifier '__read_write' can not be used for '__read_write image1d_t' prior to OpenCL version 2.0}}
#endif

#if __OPENCL_C_VERSION__ >= 200
void myPipeWrite(write_only pipe int); // expected-note {{passing argument to parameter here}}
kernel void k14(read_only pipe int p) {
  myPipeWrite(p); // expected-error {{passing '__private read_only pipe int' to parameter of incompatible type 'write_only pipe int'}}
}
#endif

#if __OPENCL_C_VERSION__ < 200
kernel void test_image3d_wo(write_only image3d_t img) {} // expected-error {{use of type '__write_only image3d_t' requires cl_khr_3d_image_writes support}}
#endif

#if __OPENCL_C_VERSION__ >= 200
kernel void read_write_twice_typedef(read_write img1d_rw i){} // expected-warning {{duplicate 'read_write' declaration specifier}}
// expected-note@-74 {{previously declared 'read_write' here}}

kernel void pipe_ro_twice(read_only read_only pipe int i){} // expected-warning{{duplicate 'read_only' declaration specifier}}
// Conflicting access qualifiers
kernel void pipe_ro_twice_tw(read_write read_only read_only pipe int i){} // expected-error{{access qualifier 'read_write' can not be used for 'read_only pipe int'}}
kernel void pipe_ro_wo(read_only write_only pipe int i){} // expected-error{{multiple access qualifiers}}

typedef read_only pipe int ROPipeInt;
kernel void pipe_ro_twice_typedef(read_only ROPipeInt i){} // expected-warning{{duplicate 'read_only' declaration specifier}}
// expected-note@-2 {{previously declared 'read_only' here}}

kernel void pass_ro_typedef_to_wo(ROPipeInt p) {
  myPipeWrite(p); // expected-error {{passing '__private ROPipeInt' (aka '__private read_only pipe int') to parameter of incompatible type 'write_only pipe int'}}
  // expected-note@-25 {{passing argument to parameter here}}
}
#endif

kernel void read_only_twice_typedef(__read_only img1d_ro i){} // expected-warning {{duplicate '__read_only' declaration specifier}}
// expected-note@-95 {{previously declared 'read_only' here}}

kernel void read_only_twice_default(read_only img1d_ro_default img){} // expected-warning {{duplicate 'read_only' declaration specifier}}
// expected-note@-101 {{previously declared 'read_only' here}}

kernel void image_wo_twice(write_only __write_only image1d_t i){} // expected-warning {{duplicate '__write_only' declaration specifier}}
kernel void image_wo_twice_typedef(write_only img1d_wo i){} // expected-warning {{duplicate 'write_only' declaration specifier}}
// expected-note@-103 {{previously declared 'write_only' here}}

