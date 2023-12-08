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
  __attribute__((visibility("hidden")))  __attribute__((aligned)) enum class EC {}; // expected-warning{{attribute 'visibility' is ignored, place it after "enum class" to apply attribute to type declaration}} \
  // expected-warning{{attribute 'aligned' is ignored, place it after "enum class" to apply attribute to type declaration}}
  __attribute__((visibility("hidden")))  __attribute__((aligned)) enum struct ES {}; // expected-warning{{attribute 'visibility' is ignored, place it after "enum struct" to apply attribute to type declaration}} \
  // expected-warning{{attribute 'aligned' is ignored, place it after "enum struct" to apply attribute to type declaration}}

  // Test that we get the same warnings for type declarations nested in a record.
  struct X {
    __attribute__((visibility("hidden")))  __attribute__((aligned)) class A; // expected-warning{{attribute 'visibility' is ignored, place it after "class" to apply attribute to type declaration}} \
    // expected-warning{{attribute 'aligned' is ignored, place it after "class" to apply attribute to type declaration}}
    __attribute__((visibility("hidden")))  __attribute__((aligned)) struct B; // expected-warning{{attribute 'visibility' is ignored, place it after "struct" to apply attribute to type declaration}} \
    // expected-warning{{attribute 'aligned' is ignored, place it after "struct" to apply attribute to type declaration}}
    __attribute__((visibility("hidden")))  __attribute__((aligned)) union C; // expected-warning{{attribute 'visibility' is ignored, place it after "union" to apply attribute to type declaration}} \
    // expected-warning{{attribute 'aligned' is ignored, place it after "union" to apply attribute to type declaration}}
    __attribute__((visibility("hidden")))  __attribute__((aligned)) enum D {D}; // expected-warning{{attribute 'visibility' is ignored, place it after "enum" to apply attribute to type declaration}} \
    // expected-warning{{attribute 'aligned' is ignored, place it after "enum" to apply attribute to type declaration}}
    __attribute__((visibility("hidden")))  __attribute__((aligned)) enum class EC {}; // expected-warning{{attribute 'visibility' is ignored, place it after "enum class" to apply attribute to type declaration}} \
    // expected-warning{{attribute 'aligned' is ignored, place it after "enum class" to apply attribute to type declaration}}
    __attribute__((visibility("hidden")))  __attribute__((aligned)) enum struct ES {}; // expected-warning{{attribute 'visibility' is ignored, place it after "enum struct" to apply attribute to type declaration}} \
    // expected-warning{{attribute 'aligned' is ignored, place it after "enum struct" to apply attribute to type declaration}}
  
    // Also test [[]] attribute syntax. (On a non-nested declaration, these
    // generate a hard "misplaced attributes" error, which we test for
    // elsewhere.)
    [[gnu::visibility("hidden")]]  [[gnu::aligned]] class E; // expected-warning{{attribute 'visibility' is ignored, place it after "class" to apply attribute to type declaration}} \
    // expected-warning{{attribute 'aligned' is ignored, place it after "class" to apply attribute to type declaration}}
    [[gnu::visibility("hidden")]]  [[gnu::aligned]] struct F; // expected-warning{{attribute 'visibility' is ignored, place it after "struct" to apply attribute to type declaration}} \
    // expected-warning{{attribute 'aligned' is ignored, place it after "struct" to apply attribute to type declaration}}
    [[gnu::visibility("hidden")]]  [[gnu::aligned]] union G; // expected-warning{{attribute 'visibility' is ignored, place it after "union" to apply attribute to type declaration}} \
    // expected-warning{{attribute 'aligned' is ignored, place it after "union" to apply attribute to type declaration}}
    [[gnu::visibility("hidden")]]  [[gnu::aligned]] enum H {H}; // expected-warning{{attribute 'visibility' is ignored, place it after "enum" to apply attribute to type declaration}} \
    // expected-warning{{attribute 'aligned' is ignored, place it after "enum" to apply attribute to type declaration}}
    [[gnu::visibility("hidden")]]  [[gnu::aligned]] enum class I {}; // expected-warning{{attribute 'visibility' is ignored, place it after "enum class" to apply attribute to type declaration}} \
    // expected-warning{{attribute 'aligned' is ignored, place it after "enum class" to apply attribute to type declaration}}
    [[gnu::visibility("hidden")]]  [[gnu::aligned]] enum struct J {}; // expected-warning{{attribute 'visibility' is ignored, place it after "enum struct" to apply attribute to type declaration}} \
    // expected-warning{{attribute 'aligned' is ignored, place it after "enum struct" to apply attribute to type declaration}}
  };
}

namespace test2 {
  __attribute__((visibility("hidden")))  __attribute__((aligned)) class A {} a;
  __attribute__((visibility("hidden")))  __attribute__((aligned)) struct B {} b;
  __attribute__((visibility("hidden")))  __attribute__((aligned)) union C {} c;
  __attribute__((visibility("hidden")))  __attribute__((aligned)) enum D {D} d;
  __attribute__((visibility("hidden")))  __attribute__((aligned)) enum class EC {} ec;
  __attribute__((visibility("hidden")))  __attribute__((aligned)) enum struct ES {} es;

  struct X {
    __attribute__((visibility("hidden")))  __attribute__((aligned)) class A {} a;
    __attribute__((visibility("hidden")))  __attribute__((aligned)) struct B {} b;
    __attribute__((visibility("hidden")))  __attribute__((aligned)) union C {} c;
    __attribute__((visibility("hidden")))  __attribute__((aligned)) enum D {D} d;
    __attribute__((visibility("hidden")))  __attribute__((aligned)) enum class EC {} ec;
    __attribute__((visibility("hidden")))  __attribute__((aligned)) enum struct ES {} es;

    [[gnu::visibility("hidden")]]  [[gnu::aligned]] class E {} e;
    [[gnu::visibility("hidden")]]  [[gnu::aligned]] struct F {} f;
    [[gnu::visibility("hidden")]]  [[gnu::aligned]] union G {} g;
    [[gnu::visibility("hidden")]]  [[gnu::aligned]] enum H {H} h;
    [[gnu::visibility("hidden")]]  [[gnu::aligned]] enum class I {} i;
    [[gnu::visibility("hidden")]]  [[gnu::aligned]] enum struct J {} j;
  };
}
