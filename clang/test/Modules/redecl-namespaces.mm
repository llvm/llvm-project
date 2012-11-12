@__experimental_modules_import redecl_namespaces_left;
@__experimental_modules_import redecl_namespaces_right;

void test() {
  A::i;
  A::j;
  A::k;  // expected-error {{no member named 'k' in namespace 'A'}}
}

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -x objective-c++ -fmodule-cache-path %t -emit-module -fmodule-name=redecl_namespaces_left %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -x objective-c++ -fmodule-cache-path %t -emit-module -fmodule-name=redecl_namespaces_right %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -fmodule-cache-path %t -w %s -verify
