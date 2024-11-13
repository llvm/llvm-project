// RUN: %clang_cc1 -std=c++98 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++11 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++14 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++17 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++20 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++23 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++2c %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s

// expected-no-diagnostics

namespace cwg1818 { // cwg1818: 3.4
extern "C" void f() {
  // This declaration binds name 'g' in the scope of function 'f',
  // but its target scope corresponds to namespace 'cwg1818' (_N4988_.[dcl.meaning]/3.5).
  // Linkage specification of 'f' applies to 'g' per _N4988_.[dcl.link]/5.
  void g();
}
// Target scope of this declaration is naturally the one
// that corresponds to namespace 'cwg1818',
// which makes it declare the same entity
// as the previous declaration per _N4988_.[basic.link]/8,
// turning it into a redeclaration per _N4988_.[basic.def]/1.
// Then _N4988_.[dcl.link]/6 applies, making it inherit
// the (С) language linkage of the previous declaration.
void g();
} // namespace cwg1818

// Check that the former 'g' has C language linkage,
// then that the latter 'g' is considered to be a redeclaration of it,
// which would make the latter 'g' inherit C language linkage from the former 'g'.

// CHECK: LinkageSpecDecl [[LINKAGE_DECL:0x[0-9a-f]+]] {{.*}} C
// CHECK: FunctionDecl [[FIRST_DECL:0x[0-9a-f]+]] parent [[LINKAGE_DECL]] {{.*}} g 'void ()'
// CHECK: FunctionDecl {{.*}} prev [[FIRST_DECL]] {{.*}} g 'void ()'
