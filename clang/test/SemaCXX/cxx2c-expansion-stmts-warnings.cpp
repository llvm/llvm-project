// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -verify=expected,old-interp
// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -verify=expected,new-interp -fexperimental-new-constant-interpreter

// Test that checks for warnings that should be emitted in expansion statements,
// but which are suppressed if we saw an error (which is why they're in a separate
// file).

#pragma GCC diagnostic warning "-Wunused-variable"
#pragma GCC diagnostic warning "-Wunused-local-typedefs"
void unused() {
  template for (int init_stmt; int expansion_var : {0}) { // expected-warning {{unused variable 'init_stmt'}} expected-warning {{unused variable 'expansion_var'}}
    int unused_var; // expected-warning {{unused variable 'unused_var'}}
    using unused_type = int; // expected-warning {{unused type alias 'unused_type'}}
    typedef int unused_typedef; // expected-warning {{unused typedef 'unused_typedef'}}
  }
}
