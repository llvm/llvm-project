// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -I %S/Inputs/template-specialization-visibility -std=c++11 %s
//
// expected-no-diagnostics

#include "c.h"

S<int> implicit_inst_class_template;
int k1 = implicit_inst_class_template.n;

S<char> explicit_inst_class_template;
int k2 = explicit_inst_class_template.n;

#include "a.h"

T<int>::S implicit_inst_member_class_template;
int k3 = implicit_inst_member_class_template.n;

T<char>::S explicit_inst_member_class_template;
int k4 = explicit_inst_member_class_template.n;

T<int>::E implicit_inst_member_enum_template;
int k5 = decltype(implicit_inst_member_enum_template)::e;

T<char>::E explicit_inst_member_enum_template;
int k6 = decltype(explicit_inst_member_enum_template)::e;
