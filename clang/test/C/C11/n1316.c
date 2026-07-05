// RUN: %clang_cc1 -verify %s

/* WG14 N1316: Yes
 * Conversion between pointers and floating types
 */

void call_ptr(void *);  // expected-note {{passing argument to parameter here}}
void call_float(float); // expected-note {{passing argument to parameter here}}

void test(float in_f, void *in_vp) {
  float f = in_vp; // expected-error {{initializing 'float' with an expression of incompatible type 'void *'}}
  void *vp = in_f; // expected-error {{initializing 'void *' with an expression of incompatible type 'float'}}

  call_ptr(f);    // expected-error {{passing 'float' to parameter of incompatible type 'void *'}}
  call_float(vp); // expected-error {{passing 'void *' to parameter of incompatible type 'float'}}

  vp = f; // expected-error {{assigning to 'void *' from incompatible type 'float'}}
  f = vp; // expected-error {{assigning to 'float' from incompatible type 'void *'}}

  struct S {
    void *ptr;
    float flt;
  } s = { f, vp }; // expected-error {{initializing 'void *' with an expression of incompatible type 'float'}} \
                      expected-error {{initializing 'float' with an expression of incompatible type 'void *'}}
}

