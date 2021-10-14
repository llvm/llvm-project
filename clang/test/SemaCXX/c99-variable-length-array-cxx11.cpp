// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -Wvla-extension %s
struct StillPOD {
  StillPOD() = default;
};

struct StillPOD2 {
  StillPOD np;
};

struct NonPOD {
  NonPOD(int) {}
};

struct POD {
  int x;
  int y;
};

// We allow VLAs of POD types, only.
void vla(int N) { // expected-note 5{{here}}
  int array1[N]; // expected-warning{{variable length arrays are a C99 feature}} expected-note {{parameter 'N'}}
  POD array2[N]; // expected-warning{{variable length arrays are a C99 feature}} expected-note {{parameter 'N'}}
  StillPOD array3[N]; // expected-warning{{variable length arrays are a C99 feature}} expected-note {{parameter 'N'}}
  StillPOD2 array4[N][3]; // expected-warning{{variable length arrays are a C99 feature}} expected-note {{parameter 'N'}}
  NonPOD array5[N]; // expected-error{{no matching constructor for initialization of 'NonPOD [N]'}}
  // expected-warning@-1{{variable length arrays are a C99 feature}} expected-note@-1 {{parameter 'N'}}
  // expected-note@-16{{candidate constructor not viable}}
  // expected-note@-18{{candidate constructor (the implicit copy constructor) not viable}}
  // expected-note@-19{{candidate constructor (the implicit move constructor) not viable}}
}
