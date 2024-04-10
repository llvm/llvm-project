// RUN: %clang_cc1 -fsyntax-only -std=gnu11 -verify=gnu -DGNU %s
// RUN: %clang_cc1 -fsyntax-only -std=c11 -fraw-string-literals -verify=gnu -DGNU %s
// RUN: %clang_cc1 -fsyntax-only -std=c11 -verify=std %s
// RUN: %clang_cc1 -fsyntax-only -std=gnu11 -fno-raw-string-literals -verify=std %s

void f() {
  (void) R"foo()foo"; // std-error {{use of undeclared identifier 'R'}}
  (void) LR"foo()foo"; // std-error {{use of undeclared identifier 'LR'}}
  (void) uR"foo()foo"; // std-error {{use of undeclared identifier 'uR'}}
  (void) u8R"foo()foo"; // std-error {{use of undeclared identifier 'u8R'}}
  (void) UR"foo()foo"; // std-error {{use of undeclared identifier 'UR'}}
}

// gnu-error@* {{missing terminating delimiter}}
// gnu-error@* {{expected expression}}
// gnu-error@* {{expected ';' after top level declarator}}
#define R "bar"
const char* s =  R"foo(";
