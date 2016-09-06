// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only

void img2d_ro(__read_only image2d_t img) {} // expected-note{{passing argument to parameter 'img' here}} expected-note{{passing argument to parameter 'img' here}}

void imgage_access_test(image2d_t img2dro, write_only image2d_t img2dwo, image3d_t img3dro) {
  img2d_ro(img2dro);
  img2d_ro(img2dwo); // expected-error{{passing '__write_only image2d_t' to parameter of incompatible type '__read_only image2d_t'}}
  img2d_ro(img3dro); // expected-error{{passing '__read_only image3d_t' to parameter of incompatible type '__read_only image2d_t'}}
}
