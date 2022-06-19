// RUN: %clang_analyze_cc1 -std=c++14 %s -verify                  \
// RUN:   -analyzer-checker=core,unix.Malloc,debug.ExprInspection \
// RUN:   -analyzer-checker=cplusplus.StringChecker               \
// RUN:   -analyzer-config eagerly-assume=false                   \
// RUN:   -analyzer-output=text

#include "Inputs/system-header-simulator-cxx.h"

void clang_analyzer_eval(bool);
void clang_analyzer_warnIfReached();
template <typename T> void clang_analyzer_dump(T);

void free(void *ptr);

void irrelevant_std_string_ctors(const char *p) {
  std::string x1;                             // no-warning
  std::string x2(2, 'x');                     // no-warning
  std::string x3(x1, /*pos=*/2);              // no-warning
  std::string x4(x1, /*pos=*/2, /*count=*/2); // no-warning
  std::string x5(p, /*count=*/(size_t)2);     // no-warning
  // skip std::string(const char*)
  std::string x6(x1.begin(), x1.end()); // no-warning
  std::string x7(x1);                   // no-warning
  std::string x8(std::move(x1));        // no-warning
  std::string x9({'a', 'b', '\0'});     // no-warning
}

void null_cstring_parameter(const char *p) {
  clang_analyzer_eval(p == 0); // expected-warning {{UNKNOWN}} expected-note {{UNKNOWN}}
  if (!p) {
    // expected-note@-1 2 {{Assuming 'p' is null}}
    // expected-note@-2 2 {{Taking true branch}}
    clang_analyzer_eval(p == 0); // expected-warning {{TRUE}} expected-note {{TRUE}}
    std::string x(p);
    // expected-warning@-1 {{The parameter must not be null}}
    // expected-note@-2    {{The parameter must not be null}}
    clang_analyzer_warnIfReached(); // no-warning
  }
}

void null_constant_parameter() {
  std::string x((char *)0);
  // expected-warning@-1 {{The parameter must not be null}}
  // expected-note@-2    {{The parameter must not be null}}
}

void unknown_ctor_param(const char *p) {
  // Pass 'UnknownVal' to the std::string constructor.
  clang_analyzer_dump((char *)(p == 0)); // expected-warning {{Unknown}} expected-note {{Unknown}}
  std::string x((char *)(p == 0));       // no-crash, no-warning
}

void ctor_notetag_on_constraining_symbol(const char *p) {
  clang_analyzer_eval(p == 0); // expected-warning {{UNKNOWN}} expected-note {{UNKNOWN}}
  std::string x(p);            // expected-note {{Assuming the pointer is not null}}
  clang_analyzer_eval(p == 0); // expected-warning {{FALSE}} expected-note {{FALSE}}

  free((void *)p); // expected-note {{Memory is released}}
  free((void *)p);
  // expected-warning@-1 {{Attempt to free released memory}}
  // expected-note@-2    {{Attempt to free released memory}}
}

void ctor_no_notetag_symbol_already_constrained(const char *p) {
  // expected-note@+2 + {{Assuming 'p' is non-null}}
  // expected-note@+1 + {{Taking false branch}}
  if (!p)
    return;

  clang_analyzer_eval(p == 0); // expected-warning {{FALSE}} expected-note {{FALSE}}
  std::string x(p);            // no-note: 'p' is already constrained to be non-null.
  clang_analyzer_eval(p == 0); // expected-warning {{FALSE}} expected-note {{FALSE}}

  free((void *)p); // expected-note {{Memory is released}}
  free((void *)p);
  // expected-warning@-1 {{Attempt to free released memory}}
  // expected-note@-2    {{Attempt to free released memory}}
}

void ctor_no_notetag_if_not_interesting(const char *p1, const char *p2) {
  std::string s1(p1); // expected-note {{Assuming the pointer is not null}}
  std::string s2(p2); // no-note: s2 is not interesting

  free((void *)p1); // expected-note {{Memory is released}}
  free((void *)p1);
  // expected-warning@-1 {{Attempt to free released memory}}
  // expected-note@-2    {{Attempt to free released memory}}
}
