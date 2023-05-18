// RUN: rm -rf %t
// RUN: %clang_cc1 -emit-pch -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -o %t.pch -I %S/Inputs -x objective-c-header %S/Inputs/pch-import-module-with-macro.pch
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -fsyntax-only -I %S/Inputs -include-pch %t.pch %s -verify
// expected-no-diagnostics

int test(int x) {
  return my_fabs(x) + fabs(x);
}

