// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -Wno-unused-value -verify %s

void basic(int * x) {    // expected-warning{{'x' is an unsafe pointer used for buffer access}}
  int *p1 = new int[10]; // not to warn
  int *p2 = new int[10]; // expected-warning{{'p2' is an unsafe pointer used for buffer access}}

#pragma clang unsafe_buffer_usage begin
  p1[5];  // not to warn

#define _INCLUDE_NO_WARN
#include "warn-unsafe-buffer-usage-pragma.h" // increment p1 in header

  int *p3 = new int[10]; // expected-warning{{'p3' is an unsafe pointer used for buffer access}}

#pragma clang unsafe_buffer_usage end
  p2[5]; //expected-note{{used in buffer access here}}
  p3[5]; //expected-note{{used in buffer access here}}
  x++;   //expected-note{{used in pointer arithmetic here}}
#define _INCLUDE_WARN
#include "warn-unsafe-buffer-usage-pragma.h" // increment p2 in header
}


void withDiagnosticWarning() {
  int *p1 = new int[10]; // not to warn
  int *p2 = new int[10]; // expected-warning{{'p2' is an unsafe pointer used for buffer access}}

  // diagnostics in opt-out region
#pragma clang unsafe_buffer_usage begin
  p1[5];  // not to warn
  p2[5];  // not to warn
#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wunsafe-buffer-usage"
  p1[5];  // not to warn
  p2[5];  // not to warn
#pragma clang diagnostic warning "-Weverything"
  p1[5];  // not to warn expected-warning{{expression result unused}}
  p2[5];  // not to warn expected-warning{{expression result unused}}
#pragma clang diagnostic pop
#pragma clang unsafe_buffer_usage end

  // opt-out region under diagnostic warning
#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wunsafe-buffer-usage"
#pragma clang unsafe_buffer_usage begin
  p1[5];  // not to warn
  p2[5];  // not to warn
#pragma clang unsafe_buffer_usage end
#pragma clang diagnostic pop

  p2[5]; // expected-note{{used in buffer access here}}
}


void withDiagnosticIgnore() {
  int *p1 = new int[10]; // not to warn
  int *p2 = new int[10]; // expected-warning{{'p2' is an unsafe pointer used for buffer access}}
  int *p3 = new int[10]; // expected-warning{{'p3' is an unsafe pointer used for buffer access}}

#pragma clang unsafe_buffer_usage begin
  p1[5];  // not to warn
  p2[5];  // not to warn
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
  p1[5];  // not to warn
  p2[5];  // not to warn
#pragma clang diagnostic ignored "-Weverything"
  p1[5];  // not to warn
  p2[5];  // not to warn
#pragma clang diagnostic pop
#pragma clang unsafe_buffer_usage end

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#pragma clang unsafe_buffer_usage begin
  p1[5];  // not to warn
  p2[5];  // not to warn
#pragma clang unsafe_buffer_usage end
#pragma clang diagnostic pop

  p2[5]; // expected-note{{used in buffer access here}}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#pragma clang unsafe_buffer_usage begin
  p1[5];  // not to warn
  p2[5];  // not to warn
#pragma clang unsafe_buffer_usage end
  p3[5];  // expected-note{{used in buffer access here}}
#pragma clang diagnostic pop
}

void noteGoesWithVarDeclWarning() {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
  int *p = new int[10]; // not to warn
#pragma clang diagnostic pop

  p[5]; // not to note since the associated warning is suppressed
}
