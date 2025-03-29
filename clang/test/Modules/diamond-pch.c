// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -emit-module -fmodules-cache-path=%t -fmodule-name=diamond_top %S/Inputs/module.modulemap
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -emit-module -fmodules-cache-path=%t -fmodule-name=diamond_left %S/Inputs/module.modulemap
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -emit-module -fmodules-cache-path=%t -fmodule-name=diamond_right %S/Inputs/module.modulemap
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -emit-module -fmodules-cache-path=%t -fmodule-name=diamond_bottom %S/Inputs/module.modulemap
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -emit-pch -fmodules-cache-path=%t -I %S/Inputs -o %t.pch %S/Inputs/diamond.h
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -fmodules-cache-path=%t -include-pch %t.pch %s -I %S/Inputs -verify
// FIXME: When we have a syntax for modules in C, use that.

void test_diamond(int i, float f, double d, char c) {
  top(&i);
  left(&f);
  right(&d);
  bottom(&c);
  bottom(&d);
  // expected-warning@-1{{incompatible pointer types passing 'double *' to parameter of type 'char *'}}
  // expected-note@Inputs/diamond_bottom.h:4{{passing argument to parameter 'x' here}}

  // Names in multiple places in the diamond.
  top_left(&c);

  left_and_right(&i);
  struct left_and_right lr;
  lr.left = 17;
}
