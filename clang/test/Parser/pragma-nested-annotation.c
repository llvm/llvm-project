// RUN: %clang_cc1 -fsyntax-only -fms-extensions -verify=expected,c %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -fms-extensions -verify=expected,cxx %s 2>&1 | FileCheck %s

// Nested _Pragma annotations must be rejected without querying identifier
// information on the annotation token. Existing recovery can leave
// inner pragma arguments in the token stream; expect the resulting errors and
// use empty declarations to keep the cases independent.
#pragma strict_gs_check(push, _Pragma("pack(1)")) // expected-warning {{unknown action for '#pragma strict_gs_check' - ignored}}
#pragma optimize("", _Pragma("weak foobar")) // expected-warning {{unexpected argument '' to '#pragma optimize'; expected 'on' or 'off'}}
#pragma optimize("", 42) // expected-warning {{unexpected argument '42' to '#pragma optimize'; expected 'on' or 'off'}}
#pragma vtordisp(_Pragma("weak foobar")) // expected-warning {{unknown action for '#pragma vtordisp' - ignored}}
// c-error@-1 {{type specifier missing}} c-error@-1 {{expected ';' after top level declarator}}
// cxx-error@-2 {{unknown type name 'foobar'}} cxx-error@-2 {{expected unqualified-id}}
;
#pragma vtordisp(push, _Pragma("weak foobar")) // expected-warning {{unknown action for '#pragma vtordisp' - ignored}}
// c-error@-1 {{type specifier missing}} c-error@-1 {{expected ';' after top level declarator}}
// cxx-error@-2 {{unknown type name 'foobar'}} cxx-error@-2 {{expected unqualified-id}}
;
#pragma pointers_to_members(_Pragma("weak foobar")) // expected-warning {{expected identifier in '#pragma pointers_to_members' - ignored}}
// c-error@-1 {{type specifier missing}} c-error@-1 {{expected ';' after top level declarator}}
// cxx-error@-2 {{unknown type name 'foobar'}} cxx-error@-2 {{expected unqualified-id}}
;
#pragma pointers_to_members(full_generality, _Pragma("weak foobar")) // expected-error {{unexpected <annot_pragma_weak>, expected to see one of 'single_inheritance', 'multiple_inheritance', or 'virtual_inheritance'}}
// c-error@-1 {{type specifier missing}} c-error@-1 {{expected ';' after top level declarator}}
// cxx-error@-2 {{unknown type name 'foobar'}} cxx-error@-2 {{expected unqualified-id}}
;
#pragma warning(_Pragma("weak foobar")) // expected-warning {{#pragma warning expected 'push', 'pop', 'default', 'disable', 'error', 'once', 'suppress', 1, 2, 3, or 4}}
// c-error@-1 {{type specifier missing}} c-error@-1 {{expected ';' after top level declarator}}
// cxx-error@-2 {{unknown type name 'foobar'}} cxx-error@-2 {{expected unqualified-id}}
;
#pragma warning(disable : 1; _Pragma("weak foobar")) // expected-warning {{#pragma warning expected 'push', 'pop', 'default', 'disable', 'error', 'once', 'suppress', 1, 2, 3, or 4}}
// c-error@-1 {{type specifier missing}} c-error@-1 {{expected ';' after top level declarator}}
// cxx-error@-2 {{unknown type name 'foobar'}} cxx-error@-2 {{expected unqualified-id}}
;
#pragma execution_character_set(_Pragma("weak foobar")) // expected-warning {{#pragma execution_character_set expected 'push' or 'pop'}}
// c-error@-1 {{type specifier missing}} c-error@-1 {{expected ';' after top level declarator}}
// cxx-error@-2 {{unknown type name 'foobar'}} cxx-error@-2 {{expected unqualified-id}}
;

void loop_state(void) {
#pragma clang loop vectorize(_Pragma("weak foobar")) // expected-error {{invalid argument; expected 'enable', 'assume_safety' or 'disable'}}
  for (int i = 0; i < 4; ++i) {}
}

void loop_width(void) {
#pragma clang loop vectorize_width(_Pragma("weak foobar")) // expected-error {{expected expression}} expected-note {{vectorize_width loop hint malformed}} expected-warning {{extra tokens at end of '#pragma clang loop vectorize_width' - ignored}}
  for (int i = 0; i < 4; ++i) {}
}

void loop_width_second_arg(void) {
#pragma clang loop vectorize_width(4, _Pragma("weak foobar")) // expected-error {{vectorize_width loop hint malformed}} expected-warning {{extra tokens at end of '#pragma clang loop vectorize_width' - ignored}}
  for (int i = 0; i < 4; ++i) {}
}

// Preserve the non-identifier alternatives in valid pragmas.
#pragma vtordisp(2)
#pragma warning(default : 1; 2 : 2)
#pragma GCC visibility push(default)
#pragma GCC visibility pop
#pragma GCC visibility push(protected)
#pragma GCC visibility pop

// __pragma captures expanded tokens, so even LexUnexpandedToken can return
// annotations to the namespace dispatch and pragma handlers.
__pragma(GCC _Pragma("weak foobar") ivdep)
__pragma(GCC visibility _Pragma("weak foobar")) // expected-warning {{expected identifier in '#pragma visibility' - ignored}}
__pragma(GCC visibility push(_Pragma("weak foobar"))) // expected-warning {{expected identifier in '#pragma visibility' - ignored}}
__pragma(clang arc_cf_code_audited _Pragma("weak foobar")) // expected-error {{expected 'begin' or 'end'}}
__pragma(clang assume_nonnull _Pragma("weak foobar")) // expected-error {{expected 'begin' or 'end'}}
__pragma(clang __set_pp_state _Pragma("weak foobar")) // expected-error {{expected identifier after '#pragma clang __set_pp_state'}}
__pragma(clang __debug macro _Pragma("weak foobar")) // expected-warning {{missing argument to debug command 'macro'}}
__pragma(clang __debug module_lookup _Pragma("weak foobar")) // expected-warning {{missing argument to debug command 'module_lookup'}}
__pragma(clang __debug modules _Pragma("weak foobar")) // expected-warning {{missing argument to debug command 'modules'}}

// Keywords and aliases must retain their original macro names.
#define const 1
#pragma clang __debug macro const
// CHECK-LABEL: MacroState {{.*}} const{{$}}
// CHECK: #define <macro> 1{{$}}
#undef const

#define __const__ 2
#pragma clang __debug macro __const__
// CHECK-LABEL: MacroState {{.*}} __const__{{$}}
// CHECK: #define <macro> 2{{$}}
#undef __const__

// C++ operator keywords carry identifier information despite punctuator kinds.
#define and 3
#pragma clang __debug macro and
// CHECK-LABEL: MacroState {{.*}} and{{$}}
// CHECK: #define <macro> 3{{$}}
#undef and

#define or 4
#pragma clang __debug macro or
// CHECK-LABEL: MacroState {{.*}} or{{$}}
// CHECK: #define <macro> 4{{$}}
#undef or

// A self-referential macro preserves the keyword through __pragma expansion.
#define int int
__pragma(clang __debug macro int)
// CHECK-LABEL: MacroState {{.*}} int{{$}}
// CHECK: #define <macro> int{{$}}
#undef int

// Punctuators must not be treated as operator keyword names.
#pragma clang __debug macro && // expected-warning {{missing argument to debug command 'macro'}}
#pragma clang __debug macro || // expected-warning {{missing argument to debug command 'macro'}}

int after_pragmas;
int use_after_pragmas(void) { return after_pragmas; }
