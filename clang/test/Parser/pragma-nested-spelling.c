// RUN: %clang_cc1 -fsyntax-only -fms-extensions -verify=expected,c %s
// RUN: %clang_cc1 -fsyntax-only -fms-extensions -fms-compatibility -verify=expected,c %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -fms-extensions -verify=expected,cxx %s

// Avoid requesting the spelling of annotation tokens in diagnostics.
// Expect existing recovery errors and use empty declarations to isolate cases.
#pragma clang optimize _Pragma("weak foobar") on // expected-error {{unexpected argument '' to '#pragma clang optimize'; expected 'on' or 'off'}}
// expected-error@-1 {{unknown type name 'foobar'}} expected-error@-1 {{expected ';' after top level declarator}}
;
#pragma clang optimize on _Pragma("weak foobar") // expected-error {{unexpected extra argument '' to '#pragma clang optimize'}}
// expected-error@-1 {{unknown type name 'foobar'}} c-error@-1 {{expected identifier or '('}} cxx-error@-1 {{expected unqualified-id}}
;
#pragma clang fp contract(_Pragma("weak foobar")) // expected-error {{unexpected argument '' to '#pragma clang fp contract'; expected 'fast' or 'on' or 'off'}}
// c-error@-1 {{type specifier missing}} c-error@-1 {{expected ';' after top level declarator}}
// cxx-error@-2 {{unknown type name 'foobar'}} cxx-error@-2 {{expected unqualified-id}}
;

// Ordinary invalid tokens retain their spelling in the diagnostics.
#pragma clang optimize 42 // expected-error {{unexpected argument '42' to '#pragma clang optimize'; expected 'on' or 'off'}}
#pragma clang optimize on 42 // expected-error {{unexpected extra argument '42' to '#pragma clang optimize'}}
#pragma clang fp contract(42) // expected-error {{unexpected argument '42' to '#pragma clang fp contract'; expected 'fast' or 'on' or 'off'}}

// __pragma expands and captures annotations before invoking the handler.
__pragma(clang optimize _Pragma("weak foobar") on) // expected-error {{unexpected argument '' to '#pragma clang optimize'; expected 'on' or 'off'}}
__pragma(clang optimize on _Pragma("weak foobar")) // expected-error {{unexpected extra argument '' to '#pragma clang optimize'}}
__pragma(clang fp contract(_Pragma("weak foobar"))) // expected-error {{unexpected argument '' to '#pragma clang fp contract'; expected 'fast' or 'on' or 'off'}}

// Preserve the keyword alternative accepted by eval_method.
#pragma clang fp eval_method(double)

int after_pragmas;
int use_after_pragmas(void) { return after_pragmas; }
