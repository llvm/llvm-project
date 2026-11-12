// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

template <bool... vals>
void f() __attribute((diagnose_if(vals, "message", "error"))) { // expected-error {{expression contains unexpanded parameter pack 'vals'}}
  [] () __attribute((diagnose_if(vals, "message", "error"))) {}(); // expected-error {{expression contains unexpanded parameter pack 'vals'}}
  [] () __attribute((diagnose_if(vals..., "message", "error"))) {}(); // expected-error {{attribute 'diagnose_if' does not support argument pack expansion}}
  [] <bool ...inner> () __attribute((diagnose_if(inner, "message", "error"))) {}(); // expected-error {{expression contains unexpanded parameter pack 'inner'}}
  ([] <bool ...inner> () __attribute((diagnose_if(inner, "message", "error"))) {}(), ...); // expected-error {{expression contains unexpanded parameter pack 'inner'}} \
                                                                                           // expected-error {{pack expansion does not contain any unexpanded parameter packs}}

  // This is fine, so check that we're actually emitting an error
  // due to the 'diagnose_if'.
  ([] () __attribute((diagnose_if(vals, "foobar", "error"))) {}(), ...); // expected-error {{foobar}} expected-note {{from 'diagnose_if'}}
}

void g() {
  f<>();
  f<false>();
  f<true, true>(); // expected-note {{in instantiation of}}
}
