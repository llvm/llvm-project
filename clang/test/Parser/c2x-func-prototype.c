// RUN: %clang_cc1 -fsyntax-only -verify=c2x -std=c2x %s
// RUN: %clang_cc1 -Wno-strict-prototypes -fsyntax-only -verify -std=c17 %s
// expected-no-diagnostics

// Functions with an identifier list are not supported in C2x.
void ident_list(a) // c2x-error {{expected ';' after top level declarator}} \
                      c2x-error {{unknown type name 'a'}}
  int a;
{}                 // c2x-error {{expected identifier or '('}}

// Functions with an empty parameter list are supported as though the function
// was declared with a parameter list of (void). Ensure they still parse.
void no_param_decl();
void no_param_defn() {}
void (*var_of_type_with_no_param)();
typedef void fn();
