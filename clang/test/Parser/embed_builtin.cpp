// RUN: %clang_cc1 %s -fsyntax-only -embed-dir=%S/Inputs -verify

void parsing_diags() {
  __builtin_pp_embed;                   // expected-error {{expected '(' after '__builtin_pp_embed'}}
  __builtin_pp_embed(;                  // expected-error {{expected a type}}
  __builtin_pp_embed();                 // expected-error {{expected a type}}
  __builtin_pp_embed(12);               // expected-error {{expected a type}}
  __builtin_pp_embed(int);              // expected-error {{expected ','}}
  __builtin_pp_embed(int, 12);          // expected-error {{expected string literal as the 2nd argument}}
  __builtin_pp_embed(int, "", 12);      // expected-error {{expected string literal as the 3rd argument}}
  __builtin_pp_embed(int, "", "", 12);  // expected-error {{expected ')'}}
  (void)__builtin_pp_embed(char, L"", "");    // expected-warning {{encoding prefix 'L' on an unevaluated string literal has no effect and is incompatible with c++2c}}
  (void)__builtin_pp_embed(char, "", L"");    // expected-warning {{encoding prefix 'L' on an unevaluated string literal has no effect and is incompatible with c++2c}}
}
