// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=override,reorder -Werror=c99-designator %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=override -Wno-reorder-init-list -Werror=initializer-overrides %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=reorder -Wno-initializer-overrides -Werror=reorder-init-list %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=good -Wno-c99-designator %s
// good-no-diagnostics

// Ensure that -Wc99-designator controls both -Winitializer-overrides and
// -Wreorder-init-list.

struct X {
  int a;
  int b;
};

void test() {
  X x{.a = 0,  // override-note {{previous initialization is here}}
      .a = 1}; // override-error {{initializer overrides prior initialization of this subobject}}
  X y{.b = 0,  // reorder-note {{previous initialization for field 'b' is here}}
      .a = 1}; // reorder-error {{ISO C++ requires field designators to be specified in declaration order; field 'b' will be initialized after field 'a'}}
}

