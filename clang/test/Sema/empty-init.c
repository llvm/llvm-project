// RUN: %clang_cc1 %s -std=c2x -Wall -pedantic -fsyntax-only -verify=good
// RUN: %clang_cc1 %s -std=c2x -Wpre-c2x-compat -fsyntax-only -verify=c2x
// RUN: %clang_cc1 %s -std=c2x -Wpre-c2x-compat -Wno-gnu-empty-initializer -fsyntax-only -verify=c2x
// RUN: %clang_cc1 %s -std=c2x -Wgnu-empty-initializer -fsyntax-only -verify=good
// RUN: %clang_cc1 %s -std=c17 -Wall -pedantic -fsyntax-only -verify=c2x-ext
// RUN: %clang_cc1 %s -std=c17 -Wgnu-empty-initializer -fsyntax-only -verify=good
// RUN: %clang_cc1 %s -std=c17 -Wc2x-extensions -fsyntax-only -verify=c2x-ext
// RUN: %clang_cc1 %s -std=c17 -Wpre-c2x-compat -fsyntax-only -verify=good

// good-no-diagnostics

// Empty brace initialization used to be a GNU extension, but the feature was
// added to C2x. We now treat empty initialization as a C extension rather than
// a GNU extension. Thus, -Wgnu-empty-initializer is always silently ignored.

struct S {
  int a;
};

struct S s = {};     /* c2x-warning {{use of an empty initializer is incompatible with C standards before C23}}
                        c2x-ext-warning {{use of an empty initializer is a C23 extension}}
                      */

void func(void) {
  struct S s2 = {};  /* c2x-warning {{use of an empty initializer is incompatible with C standards before C23}}
                        c2x-ext-warning {{use of an empty initializer is a C23 extension}}
                      */
  (void)s2;
}

