// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -Wimplicit-fallthrough -DCLANG_PREFIX -DCOMMAND_LINE_FALLTHROUGH=[[clang::fallthrough]] -DUNCHOSEN=[[fallthrough]] %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -Wimplicit-fallthrough -DCOMMAND_LINE_FALLTHROUGH=[[fallthrough]] %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++1z -Wimplicit-fallthrough -DCLANG_PREFIX -DCOMMAND_LINE_FALLTHROUGH=[[clang::fallthrough]] %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++1z -Wimplicit-fallthrough -DCOMMAND_LINE_FALLTHROUGH=[[clang::fallthrough]] %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++1z -Wimplicit-fallthrough -DCOMMAND_LINE_FALLTHROUGH=[[fallthrough]] -DUNCHOSEN=[[clang::fallthrough]] %s

int fallthrough_compatibility_macro_from_command_line(int n) {
  switch (n) {
    case 0:
      n = n * 10;
    case 1: // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert 'COMMAND_LINE_FALLTHROUGH;' to silence this warning}} expected-note{{insert 'break;' to avoid fall-through}}
      ;
  }
  return n;
}

#ifdef CLANG_PREFIX
#define COMPATIBILITY_FALLTHROUGH   [ [ /* test */  clang /* test */ \
    ::  fallthrough  ]  ]    // testing whitespace and comments in macro definition
#else
#define COMPATIBILITY_FALLTHROUGH   [ [ /* test */  /* test */ \
    fallthrough  ]  ]    // testing whitespace and comments in macro definition
#endif

int fallthrough_compatibility_macro_from_source(int n) {
  switch (n) {
    case 0:
      n = n * 20;
    case 1: // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert 'COMPATIBILITY_FALLTHROUGH;' to silence this warning}} expected-note{{insert 'break;' to avoid fall-through}}
      ;
  }
  return n;
}

// Deeper macro substitution
#ifdef CLANG_PREFIX
#define M1 [[clang::fallthrough]]
#else
#define M1 [[fallthrough]]
#endif
#ifdef __clang__
#define M2 M1
#else
#define M2
#endif

#define WRONG_MACRO1 clang::fallthrough
#define WRONG_MACRO2 [[clang::fallthrough]
#define WRONG_MACRO3 [[clang::fall through]]
#define WRONG_MACRO4 [[clang::fallthrough]]]

int fallthrough_compatibility_macro_in_macro(int n) {
  switch (n) {
    case 0:
      n = n * 20;
    case 1: // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert 'M1;' to silence this warning}} expected-note{{insert 'break;' to avoid fall-through}}
                                                                          // there was an idea that this ^ should be M2
      ;
  }
  return n;
}

#undef M1
#undef M2
#undef COMPATIBILITY_FALLTHROUGH
#undef COMMAND_LINE_FALLTHROUGH
#undef UNCHOSEN

int fallthrough_compatibility_macro_undefined(int n) {
  switch (n) {
    case 0:
      n = n * 20;
#if __cplusplus <= 201402L
    case 1: // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert '[[clang::fallthrough]];' to silence this warning}} expected-note{{insert 'break;' to avoid fall-through}}
#else
    case 1: // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert '[[fallthrough]];' to silence this warning}} expected-note{{insert 'break;' to avoid fall-through}}
#endif
      ;
  }
#define TOO_LATE [[clang::fallthrough]]
  return n;
}
#undef TOO_LATE

#define MACRO_WITH_HISTORY 11111111
#undef MACRO_WITH_HISTORY
#define MACRO_WITH_HISTORY [[clang::fallthrough]]
#undef MACRO_WITH_HISTORY
#define MACRO_WITH_HISTORY 2222222

int fallthrough_compatibility_macro_history(int n) {
  switch (n) {
    case 0:
      n = n * 20;
#undef MACRO_WITH_HISTORY
#if __cplusplus <= 201402L
    case 1: // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert '[[clang::fallthrough]];' to silence this warning}} expected-note{{insert 'break;' to avoid fall-through}}
#else
    case 1: // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert '[[fallthrough]];' to silence this warning}} expected-note{{insert 'break;' to avoid fall-through}}
#endif
      ;
#define MACRO_WITH_HISTORY [[clang::fallthrough]]
  }
  return n;
}

#undef MACRO_WITH_HISTORY
#define MACRO_WITH_HISTORY 11111111
#undef MACRO_WITH_HISTORY
#define MACRO_WITH_HISTORY [[clang::fallthrough]]
#undef MACRO_WITH_HISTORY
#define MACRO_WITH_HISTORY 2222222
#undef MACRO_WITH_HISTORY

int fallthrough_compatibility_macro_history2(int n) {
  switch (n) {
    case 0:
      n = n * 20;
#define MACRO_WITH_HISTORY [[clang::fallthrough]]
    case 1: // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert 'MACRO_WITH_HISTORY;' to silence this warning}} expected-note{{insert 'break;' to avoid fall-through}}
      ;
#undef MACRO_WITH_HISTORY
#define MACRO_WITH_HISTORY 3333333
#undef MACRO_WITH_HISTORY
#define MACRO_WITH_HISTORY 4444444
#undef MACRO_WITH_HISTORY
#define MACRO_WITH_HISTORY 5555555
  }
  return n;
}

template<const int N>
int fallthrough_compatibility_macro_history_template(int n) {
  switch (N * n) {
    case 0:
      n = n * 20;
#define MACRO_WITH_HISTORY2 [[clang::fallthrough]]
    case 1: // expected-warning{{unannotated fall-through between switch labels}} expected-note{{insert 'MACRO_WITH_HISTORY2;' to silence this warning}} expected-note{{insert 'break;' to avoid fall-through}}
      ;
#undef MACRO_WITH_HISTORY2
#define MACRO_WITH_HISTORY2 3333333
  }
  return n;
}

#undef MACRO_WITH_HISTORY2
#define MACRO_WITH_HISTORY2 4444444
#undef MACRO_WITH_HISTORY2
#define MACRO_WITH_HISTORY2 5555555

void f() {
  fallthrough_compatibility_macro_history_template<1>(0); // expected-note{{in instantiation of function template specialization 'fallthrough_compatibility_macro_history_template<1>' requested here}}
}
