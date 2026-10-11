// RUN: %clang_cc1 -fsyntax-only -fms-extensions -verify %s
// RUN: %clang_cc1 -fsyntax-only -fms-extensions -fms-compatibility -verify %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -fms-extensions -verify %s

// Annotation arguments have no spelling. Omit the empty quoted argument and
// consume the remaining directive, so subsequent declarations remain intact.
#pragma clang optimize _Pragma("weak foobar") on // expected-error {{unexpected argument to '#pragma clang optimize'; expected 'on' or 'off'}}
#pragma clang optimize on _Pragma("weak foobar") // expected-error {{unexpected extra argument to '#pragma clang optimize'}}
#pragma clang fp contract(_Pragma("weak foobar")) // expected-error {{unexpected argument to '#pragma clang fp contract'; expected 'fast' or 'on' or 'off'}}
#pragma optimize("", _Pragma("weak foobar")) // expected-warning {{unexpected argument to '#pragma optimize'; expected 'on' or 'off'}}

#pragma vtordisp(_Pragma("weak foobar")) // expected-warning {{unknown action for '#pragma vtordisp' - ignored}}
#pragma pointers_to_members(_Pragma("weak foobar")) // expected-warning {{expected identifier in '#pragma pointers_to_members' - ignored}}
#pragma pointers_to_members(full_generality, _Pragma("weak foobar")) // expected-error {{unexpected <annot_pragma_weak>, expected to see one of 'single_inheritance', 'multiple_inheritance', or 'virtual_inheritance'}}
#pragma warning(_Pragma("weak foobar")) // expected-warning {{#pragma warning expected 'push', 'pop', 'default', 'disable', 'error', 'once', 'suppress', 1, 2, 3, or 4}}
#pragma execution_character_set(_Pragma("weak foobar")) // expected-warning {{#pragma execution_character_set expected 'push' or 'pop'}}

// Ordinary invalid arguments retain their original spelling.
#pragma clang optimize 42 // expected-error {{unexpected argument '42' to '#pragma clang optimize'; expected 'on' or 'off'}}
#pragma clang optimize on 42 // expected-error {{unexpected extra argument '42' to '#pragma clang optimize'}}
#pragma clang fp contract(42) // expected-error {{unexpected argument '42' to '#pragma clang fp contract'; expected 'fast' or 'on' or 'off'}}
#pragma optimize("", 42) // expected-warning {{unexpected argument '42' to '#pragma optimize'; expected 'on' or 'off'}}

// __debug dump leaves its arguments and eod for the parser. Its inner eod
// must not terminate the enclosing pragma's recovery or argument capture.
#pragma vtordisp(_Pragma("clang __debug dump x")) // expected-warning {{unknown action for '#pragma vtordisp' - ignored}}
int after_vtordisp;
#pragma optimize("", _Pragma("clang __debug dump x")) // expected-warning {{unexpected argument to '#pragma optimize'; expected 'on' or 'off'}}
int after_ms_optimize;
__pragma(optimize("", _Pragma("clang __debug dump x"))) // expected-warning {{unexpected argument to '#pragma optimize'; expected 'on' or 'off'}}
int after_ms_capture;

// Captured __pragma token streams have their own eod too.
#pragma vtordisp(__pragma(clang __debug dump x)) // expected-warning {{unknown action for '#pragma vtordisp' - ignored}}
int after_nested_ms_pragma;
__pragma(optimize("", __pragma(clang __debug dump x))) // expected-warning {{unexpected argument to '#pragma optimize'; expected 'on' or 'off'}}
int after_nested_ms_capture;

// Both captures start in the same macro, then finish after its lexer is popped.
#define BEGIN_NESTED_PRAGMA __pragma(optimize("", __pragma(clang __debug dump
BEGIN_NESTED_PRAGMA x))) // expected-warning {{unexpected argument to '#pragma optimize'; expected 'on' or 'off'}}
#undef BEGIN_NESTED_PRAGMA
int after_macro;

int use_after_pragmas(void) {
  return after_vtordisp + after_ms_optimize + after_ms_capture +
         after_nested_ms_pragma + after_nested_ms_capture + after_macro;
}
