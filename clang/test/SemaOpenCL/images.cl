// RUN: %clang_cc1 %s -cl-std=CL2.0 -verify -pedantic -fsyntax-only

void img2d_ro(read_only image2d_t); // expected-note 3{{passing argument to parameter here}}
void img2d_wo(write_only image2d_t); // expected-note 2{{passing argument to parameter here}}
void img2d_rw(read_write image2d_t); // expected-note 2{{passing argument to parameter here}}
void img2d_default(image2d_t); // expected-note 2{{passing argument to parameter here}}

void imgage_access_test(image2d_t img2dro, image3d_t img3dro) {
  img2d_ro(img2dro); // read_only = read_only
  img2d_ro(img3dro); // expected-error{{passing '__private __read_only image3d_t' to parameter of incompatible type '__read_only image2d_t'}}
}

kernel void read_only_access_test(read_only image2d_t img) {
  img2d_ro(img); // read_only = read_only
  img2d_wo(img); // expected-error {{passing '__private __read_only image2d_t' to parameter of incompatible type '__write_only image2d_t'}}
  img2d_rw(img); // expected-error {{passing '__private __read_only image2d_t' to parameter of incompatible type '__read_write image2d_t'}}
  img2d_default(img); // read_only = read_only
}

kernel void write_only_access_test(write_only image2d_t img) {
  img2d_ro(img); // expected-error {{passing '__private __write_only image2d_t' to parameter of incompatible type '__read_only image2d_t'}}
  img2d_wo(img); // write_only = write_only
  img2d_rw(img); // expected-error {{passing '__private __write_only image2d_t' to parameter of incompatible type '__read_write image2d_t'}}
  img2d_default(img); // expected-error {{passing '__private __write_only image2d_t' to parameter of incompatible type '__read_only image2d_t'}}
}

kernel void read_write_access_test(read_write image2d_t img) {
  img2d_ro(img);  // expected-error {{passing '__private __read_write image2d_t' to parameter of incompatible type '__read_only image2d_t'}}
  img2d_wo(img); // expected-error {{passing '__private __read_write image2d_t' to parameter of incompatible type '__write_only image2d_t'}}
  img2d_rw(img); //read_write = read_write
  img2d_default(img); // expected-error {{passing '__private __read_write image2d_t' to parameter of incompatible type '__read_only image2d_t'}}
}
