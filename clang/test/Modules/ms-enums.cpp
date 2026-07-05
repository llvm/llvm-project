// RUN: rm -rf %t
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions -fms-compatibility -x c++ -std=c++20 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/ms-enums %s -verify -fno-modules-error-recovery

#include "B.h"
// expected-note@A.h:1 {{declaration here is not visible}}
// expected-note@A.h:1 2{{definition here is not reachable}}

fwd_enum gv_enum; // expected-error {{must be imported}}

struct Foo {
  enum fwd_enum enum_field; // expected-error 2 {{must be imported}}
};
