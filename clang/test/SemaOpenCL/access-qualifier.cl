// RUN: %clang_cc1 -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-std=CL1.2 %s -cl-ext=-cl_khr_3d_image_writes
// RUN: %clang_cc1 -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-std=CL2.0 %s
// RUN: %clang_cc1 -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-std=CL3.0 %s
// RUN: %clang_cc1 -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-std=CL3.0 %s -cl-ext=-__opencl_c_read_write_images
// RUN: %clang_cc1 -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-std=clc++2021 %s
// RUN: %clang_cc1 -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-std=clc++2021 %s -cl-ext=-__opencl_c_read_write_images

typedef image1d_t img1d_ro_default; // expected-note {{previously declared 'read_only' here}}

typedef write_only image1d_t img1d_wo; // expected-note {{previously declared 'write_only' here}}
typedef read_only image1d_t img1d_ro;

#if (__OPENCL_C_VERSION__ == 200) || ((__OPENCL_CPP_VERSION__ == 202100 || __OPENCL_C_VERSION__ == 300) && defined(__opencl_c_read_write_images))
typedef read_write image1d_t img1d_rw;
#endif

typedef int Int;
typedef read_only int IntRO; // expected-error {{access qualifier can only be used for pipe and image type}}

void myWrite(write_only image1d_t);
#if !defined(__OPENCL_CPP_VERSION__)
// expected-note@-2 {{passing argument to parameter here}}
// expected-note@-3 {{passing argument to parameter here}}
#else
// expected-note@-5 {{candidate function not viable: no known conversion from '__private img1d_ro' (aka '__private __read_only image1d_t') to '__private __write_only image1d_t' for 1st argument}}
// expected-note@-6 {{candidate function not viable: no known conversion from '__private img1d_ro_default' (aka '__private __read_only image1d_t') to '__private __write_only image1d_t' for 1st argument}}
#endif

void myRead(read_only image1d_t);
#if !defined(__OPENCL_CPP_VERSION__)
// expected-note@-2 {{passing argument to parameter here}}
#else
// expected-note@-4 {{candidate function not viable: no known conversion from '__private img1d_wo' (aka '__private __write_only image1d_t') to '__private __read_only image1d_t' for 1st argument}}
#endif

#if (__OPENCL_C_VERSION__ == 200) || ((__OPENCL_CPP_VERSION__ == 202100 || __OPENCL_C_VERSION__ == 300) && defined(__opencl_c_read_write_images))
void myReadWrite(read_write image1d_t);
#else
void myReadWrite(read_write image1d_t); // expected-error {{access qualifier 'read_write' can not be used for '__read_write image1d_t' prior to OpenCL C version 2.0 or in version 3.0 and without __opencl_c_read_write_images feature}}
#endif


kernel void k1(img1d_wo img) {
  myRead(img);
#if !defined(__OPENCL_CPP_VERSION__)
// expected-error@-2 {{passing '__private img1d_wo' (aka '__private __write_only image1d_t') to parameter of incompatible type '__read_only image1d_t'}}
#else
// expected-error@-4 {{no matching function for call to 'myRead'}}
#endif
}

kernel void k2(img1d_ro img) {
  myWrite(img);
#if !defined(__OPENCL_CPP_VERSION__)
// expected-error@-2 {{passing '__private img1d_ro' (aka '__private __read_only image1d_t') to parameter of incompatible type '__write_only image1d_t'}}
#else
// expected-error@-4 {{no matching function for call to 'myWrite'}}
#endif
}

kernel void k3(img1d_wo img) {
  myWrite(img);
}

#if (__OPENCL_C_VERSION__ == 200) || ((__OPENCL_CPP_VERSION__ == 202100 || __OPENCL_C_VERSION__ == 300) && defined(__opencl_c_read_write_images))
kernel void k4(img1d_rw img) {
  myReadWrite(img);
}
#endif

kernel void k5(img1d_ro_default img) {
  myWrite(img);
#if !defined(__OPENCL_CPP_VERSION__)
// expected-error@-2 {{passing '__private img1d_ro_default' (aka '__private __read_only image1d_t') to parameter of incompatible type '__write_only image1d_t'}}
#else
// expected-error@-4 {{no matching function for call to 'myWrite'}}
#endif
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

#if (__OPENCL_C_VERSION__ == 200) || ((__OPENCL_CPP_VERSION__ == 202100 || __OPENCL_C_VERSION__ == 300) && defined(__opencl_c_read_write_images))
kernel void k13(read_write pipe int i){} // expected-error{{access qualifier 'read_write' can not be used for 'read_only pipe int'}}
#else
kernel void k13(__read_write image1d_t i){} // expected-error{{access qualifier '__read_write' can not be used for '__read_write image1d_t' prior to OpenCL C version 2.0 or in version 3.0 and without __opencl_c_read_write_images feature}}
#endif

#if defined(__OPENCL_C_VERSION__) && __OPENCL_C_VERSION__ < 200
kernel void test_image3d_wo(write_only image3d_t img) {} // expected-error {{use of type '__write_only image3d_t' requires cl_khr_3d_image_writes support}}
#endif

#if (__OPENCL_C_VERSION__ == 200) || ((__OPENCL_CPP_VERSION__ == 202100 || __OPENCL_C_VERSION__ == 300) && defined(__opencl_c_read_write_images))
kernel void read_write_twice_typedef(read_write img1d_rw i){} // expected-warning {{duplicate 'read_write' declaration specifier}}
// expected-note@-94 {{previously declared 'read_write' here}}
#endif

#if __OPENCL_C_VERSION__ >= 200
void myPipeWrite(write_only pipe int); // expected-note {{passing argument to parameter here}}
kernel void k14(read_only pipe int p) {
  myPipeWrite(p); // expected-error {{passing '__private read_only pipe int' to parameter of incompatible type 'write_only pipe int'}}
}

kernel void pipe_ro_twice(read_only read_only pipe int i){} // expected-warning{{duplicate 'read_only' declaration specifier}}
// Conflicting access qualifiers
kernel void pipe_ro_twice_tw(read_write read_only read_only pipe int i){} // expected-error{{access qualifier 'read_write' can not be used for 'read_only pipe int'}}
kernel void pipe_ro_wo(read_only write_only pipe int i){} // expected-error{{multiple access qualifiers}}

typedef read_only pipe int ROPipeInt;
kernel void pipe_ro_twice_typedef(read_only ROPipeInt i){} // expected-warning{{duplicate 'read_only' declaration specifier}}
// expected-note@-2 {{previously declared 'read_only' here}}

kernel void pass_ro_typedef_to_wo(ROPipeInt p) {
  myPipeWrite(p); // expected-error {{passing '__private ROPipeInt' (aka '__private read_only pipe int') to parameter of incompatible type 'write_only pipe int'}}
  // expected-note@-16 {{passing argument to parameter here}}
}
#endif

kernel void read_only_twice_typedef(__read_only img1d_ro i){} // expected-warning {{duplicate '__read_only' declaration specifier}}
// expected-note@-122 {{previously declared 'read_only' here}}

kernel void read_only_twice_default(read_only img1d_ro_default img){} // expected-warning {{duplicate 'read_only' declaration specifier}}
// expected-note@-128 {{previously declared 'read_only' here}}

kernel void image_wo_twice(write_only __write_only image1d_t i){} // expected-warning {{duplicate '__write_only' declaration specifier}}
kernel void image_wo_twice_typedef(write_only img1d_wo i){} // expected-warning {{duplicate 'write_only' declaration specifier}}
// expected-note@-130 {{previously declared 'write_only' here}}
