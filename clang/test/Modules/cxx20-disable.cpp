// RUN: rm -rf %t && mkdir %t
// RUN: %clang_cc1 -x objective-c++ -std=c++20                  -I %t %s -verify=enabled
// RUN: %clang_cc1 -x objective-c++ -std=c++20 -fno-cxx-modules -I %t %s -verify=disabled

// enabled-no-diagnostics

// The spelling of these errors is misleading.
// The important thing is Clang rejected C++20 modules syntax.
export module Foo; // disabled-error{{expected template}}
                   // disabled-error@-1{{unknown type name 'module'}}
