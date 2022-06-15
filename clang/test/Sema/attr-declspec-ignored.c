// RUN: %clang_cc1 %s -verify -fsyntax-only

__attribute__((visibility("hidden")))  __attribute__((aligned)) struct A; // expected-warning{{attribute 'visibility' is ignored, place it after "struct" to apply attribute to type declaration}} \
// expected-warning{{attribute 'aligned' is ignored, place it after "struct" to apply attribute to type declaration}}
__attribute__((visibility("hidden")))  __attribute__((aligned)) union B; // expected-warning{{attribute 'visibility' is ignored, place it after "union" to apply attribute to type declaration}} \
// expected-warning{{attribute 'aligned' is ignored, place it after "union" to apply attribute to type declaration}} 
__attribute__((visibility("hidden")))  __attribute__((aligned)) enum C {C}; // expected-warning{{attribute 'visibility' is ignored, place it after "enum" to apply attribute to type declaration}} \
// expected-warning{{attribute 'aligned' is ignored, place it after "enum" to apply attribute to type declaration}}

// Make sure that we produce the same warnings on block declarations.
void func() {
  __attribute__((visibility("hidden")))  __attribute__((aligned)) struct A; // expected-warning{{attribute 'visibility' is ignored, place it after "struct" to apply attribute to type declaration}} \
  // expected-warning{{attribute 'aligned' is ignored, place it after "struct" to apply attribute to type declaration}}
  __attribute__((visibility("hidden")))  __attribute__((aligned)) union B; // expected-warning{{attribute 'visibility' is ignored, place it after "union" to apply attribute to type declaration}} \
  // expected-warning{{attribute 'aligned' is ignored, place it after "union" to apply attribute to type declaration}}
  __attribute__((visibility("hidden")))  __attribute__((aligned)) enum C {C}; // expected-warning{{attribute 'visibility' is ignored, place it after "enum" to apply attribute to type declaration}} \
  // expected-warning{{attribute 'aligned' is ignored, place it after "enum" to apply attribute to type declaration}}
}

__attribute__((visibility("hidden")))  __attribute__((aligned)) struct D {} d;
__attribute__((visibility("hidden")))  __attribute__((aligned)) union E {} e;
__attribute__((visibility("hidden")))  __attribute__((aligned)) enum F {F} f;
