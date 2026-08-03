// RUN: %clang_cc1 -fsyntax-only -flifetime-safety-inference -Wlifetime-safety-noescape-strict -verify %s

#include "Inputs/lifetime-analysis.h"

struct [[gsl::Owner]] MyObj {
  int id;
  ~MyObj() {}
};

struct [[gsl::Pointer()]] View {
  View(const MyObj& obj [[clang::noescape]]);
};

View identity_lifetimebound(View v [[clang::lifetimebound]]) { return v; }

View escape_through_lifetimebound_call(
    const MyObj& in [[clang::noescape]]) { // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
  return identity_lifetimebound(in); // expected-note {{escapes through this call}}
}

View no_annotation_identity(View v) { return v; }

View escape_through_unannotated_call(const MyObj& in [[clang::noescape]]) { // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
  return no_annotation_identity(in); // expected-note {{escapes through this call}}
}

void escape_through_param(const MyObj& in [[clang::noescape]], // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
                          std::vector<View> &v) {
  v.push_back(in); // expected-note {{escapes through this call}}
}
