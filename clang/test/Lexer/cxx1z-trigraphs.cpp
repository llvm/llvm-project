// RUN: %clang_cc1 -std=c++1z %s -verify
// RUN: %clang_cc1 -std=c++1z %s -ftrigraphs -fsyntax-only 2>&1 | FileCheck --check-prefix=TRIGRAPHS %s

??= define foo ; // expected-error {{}} expected-warning {{trigraph ignored}}

static_assert("??="[0] == '#', ""); // expected-error {{failed}} expected-warning {{trigraph ignored}}

// ??/
error here; // expected-error {{}}

// Note, there is intentionally trailing whitespace two lines below.
// TRIGRAPHS: :[[@LINE+1]]:{{.*}} backslash and newline separated by space
// ??/  
error here; // expected-error {{}}
