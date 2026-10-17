// RUN: %clang_cc1 -fsyntax-only -fms-extensions -verify %s
// RUN: %clang_cc1 -fsyntax-only -fms-extensions -fms-compatibility -verify %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -fms-extensions -verify %s

// Nested _Pragma annotations must be rejected without querying identifier
// information or spelling on the annotation token.
#pragma strict_gs_check(push, _Pragma("pack(1)")) // expected-warning {{unknown action for '#pragma strict_gs_check' - ignored}}
#pragma optimize("", _Pragma("weak foobar")) // expected-warning {{unexpected argument to '#pragma optimize'; expected 'on' or 'off'}}
#pragma vtordisp(_Pragma("weak foobar")) // expected-warning {{unknown action for '#pragma vtordisp' - ignored}}
#pragma vtordisp(push, _Pragma("weak foobar")) // expected-warning {{unknown action for '#pragma vtordisp' - ignored}}
#pragma pointers_to_members(_Pragma("weak foobar")) // expected-warning {{expected identifier in '#pragma pointers_to_members' - ignored}}
#pragma pointers_to_members(full_generality, _Pragma("weak foobar")) // expected-error {{unexpected <annot_pragma_weak>, expected to see one of 'single_inheritance', 'multiple_inheritance', or 'virtual_inheritance'}}
#pragma warning(_Pragma("weak foobar")) // expected-warning {{#pragma warning expected 'push', 'pop', 'default', 'disable', 'error', 'once', 'suppress', 1, 2, 3, or 4}}
#pragma warning(disable : 1; _Pragma("weak foobar")) // expected-warning {{#pragma warning expected 'push', 'pop', 'default', 'disable', 'error', 'once', 'suppress', 1, 2, 3, or 4}}
#pragma execution_character_set(_Pragma("weak foobar")) // expected-warning {{#pragma execution_character_set expected 'push' or 'pop'}}

#pragma clang optimize _Pragma("weak foobar") on // expected-error {{unexpected argument to '#pragma clang optimize'; expected 'on' or 'off'}}
#pragma clang optimize on _Pragma("weak foobar") // expected-error {{unexpected extra argument to '#pragma clang optimize'}}
#pragma clang fp contract(_Pragma("weak foobar")) // expected-error {{unexpected argument to '#pragma clang fp contract'; expected 'fast' or 'on' or 'off'}}

// Ordinary invalid tokens retain their spelling in the diagnostics.
#pragma clang optimize 42 // expected-error {{unexpected argument '42' to '#pragma clang optimize'; expected 'on' or 'off'}}
#pragma clang optimize on 42 // expected-error {{unexpected extra argument '42' to '#pragma clang optimize'}}
#pragma clang fp contract(42) // expected-error {{unexpected argument '42' to '#pragma clang fp contract'; expected 'fast' or 'on' or 'off'}}
#pragma optimize("", 42) // expected-warning {{unexpected argument '42' to '#pragma optimize'; expected 'on' or 'off'}}

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
#pragma vtordisp()
#pragma vtordisp(2)
#pragma vtordisp(push, 1)
#pragma vtordisp(pop)
#pragma warning(default : 1; 2 : 2)
#pragma GCC visibility push(default)
#pragma GCC visibility pop
#pragma GCC visibility push(protected)
#pragma GCC visibility pop
#pragma clang fp eval_method(double)

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

// Macro names may be keywords.
#define int int
__pragma(clang __debug macro int)
#undef int

// Unlike most pragma handlers, __debug dump leaves its arguments and eod for
// the parser. That inner eod must not terminate the enclosing pragma, either
// during error recovery or while saving arguments for later parsing.
#pragma vtordisp(_Pragma("clang __debug dump x")) // expected-warning {{unknown action for '#pragma vtordisp' - ignored}}
int after_vtordisp;
#pragma optimize("", _Pragma("clang __debug dump x")) // expected-warning {{unexpected argument to '#pragma optimize'; expected 'on' or 'off'}}
int after_ms_optimize;

// __pragma expands the inner pragma while capturing its arguments, before the
// outer handler runs. Capturing must also ignore the inner eod.
__pragma(optimize("", _Pragma("clang __debug dump x"))) // expected-warning {{unexpected argument to '#pragma optimize'; expected 'on' or 'off'}}
int after_ms_pragmas;

// An inner __pragma can also defer parsing. Its captured token stream has a
// separate eod which must not terminate the outer directive or argument capture.
#pragma vtordisp(__pragma(clang __debug dump x)) // expected-warning {{unknown action for '#pragma vtordisp' - ignored}}
int after_nested_ms_pragma;
__pragma(optimize("", __pragma(clang __debug dump x))) // expected-warning {{unexpected argument to '#pragma optimize'; expected 'on' or 'off'}}
int after_nested_ms_capture;

// Both captures start in the same macro, but continue after it is popped.
// Keep the outer boundary updated when the inner capture finishes.
#define BEGIN_NESTED_PRAGMA __pragma(optimize("", __pragma(clang __debug dump
BEGIN_NESTED_PRAGMA x))) // expected-warning {{unexpected argument to '#pragma optimize'; expected 'on' or 'off'}}
#undef BEGIN_NESTED_PRAGMA
int after_macro;

int after_pragmas;
int use_after_pragmas(void) {
  return after_pragmas + after_vtordisp + after_ms_optimize + after_ms_pragmas +
         after_nested_ms_pragma + after_nested_ms_capture + after_macro;
}
