// RUN: %clang_cc1 %s -verify -fsyntax-only

namespace test1 {
  __attribute__((visibility("hidden")))  __attribute__((aligned)) class A; // expected-warning{{attribute 'visibility' is ignored, place it after "class" to apply attribute to type declaration}} \
  // expected-warning{{attribute 'aligned' is ignored, place it after "class" to apply attribute to type declaration}}
  __attribute__((visibility("hidden")))  __attribute__((aligned)) struct B; // expected-warning{{attribute 'visibility' is ignored, place it after "struct" to apply attribute to type declaration}} \
  // expected-warning{{attribute 'aligned' is ignored, place it after "struct" to apply attribute to type declaration}}
  __attribute__((visibility("hidden")))  __attribute__((aligned)) union C; // expected-warning{{attribute 'visibility' is ignored, place it after "union" to apply attribute to type declaration}} \
  // expected-warning{{attribute 'aligned' is ignored, place it after "union" to apply attribute to type declaration}} 
  __attribute__((visibility("hidden")))  __attribute__((aligned)) enum D {D}; // expected-warning{{attribute 'visibility' is ignored, place it after "enum" to apply attribute to type declaration}} \
  // expected-warning{{attribute 'aligned' is ignored, place it after "enum" to apply attribute to type declaration}}
}

namespace test2 {
  __attribute__((visibility("hidden")))  __attribute__((aligned)) class A {} a;
  __attribute__((visibility("hidden")))  __attribute__((aligned)) struct B {} b;
  __attribute__((visibility("hidden")))  __attribute__((aligned)) union C {} c;
  __attribute__((visibility("hidden")))  __attribute__((aligned)) enum D {D} d;
}
