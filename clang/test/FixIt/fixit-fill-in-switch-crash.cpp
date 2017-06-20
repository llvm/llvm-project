// RUN: %clang_cc1 -verify -std=c++11 %s
// RUN: not %clang_cc1 -fdiagnostics-parseable-fixits -std=c++11 %s 2>&1 | FileCheck %s

enum Color {
  Black, Red
};

void dontCrashOnEmptySubStmt(Color c) { // expected-note {{to match this '{'}}
  switch (c) { // expected-note {{to match this '{'}} \
               // expected-warning {{enumeration value 'Red' not handled in switch}} \
               // expected-note {{add missing switch cases}}
  case Black: // CHECK: fix-it:{{.*}}:{[[@LINE+3]]:10-[[@LINE+3]]:10}:"case Red:\n<#code#>\nbreak;\n"
  // expected-error@+2 {{expected expression}}
  // expected-error@+1 2 {{expected '}'}}
  case //
