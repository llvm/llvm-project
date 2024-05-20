// RUN: %clang_cc1 -fsyntax-only -std=gnu11 -verify=supported %s
// RUN: %clang_cc1 -fsyntax-only -std=c11 -fraw-string-literals -verify=supported %s
// RUN: %clang_cc1 -fsyntax-only -std=gnu89 -verify=unsupported %s
// RUN: %clang_cc1 -fsyntax-only -std=c11 -verify=unsupported %s
// RUN: %clang_cc1 -fsyntax-only -std=gnu11 -fno-raw-string-literals -verify=unsupported %s

// RUN: %clang_cc1 -x c++ -fsyntax-only -Wno-unused -std=c++03 -verify=unsupported,cxx-unsupported %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wno-unused -std=gnu++03 -verify=unsupported,cxx-unsupported %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wno-unused -std=c++03 -fraw-string-literals -verify=supported %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wno-unused -std=gnu++03 -fraw-string-literals -verify=supported %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wno-unused -std=c++11 -verify=supported,cxx %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wno-unused -std=gnu++11 -verify=supported,cxx %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wno-unused -std=c++11 -fno-raw-string-literals -verify=supported,no %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wno-unused -std=gnu++11 -fno-raw-string-literals -verify=supported,no %s

// GCC supports raw string literals in C99 and later in '-std=gnuXY' mode; we
// additionally provide '-f[no-]raw-string-literals' to enable/disable them
// explicitly in C.
//
// We do not allow disabling or enabling raw string literals in C++ mode if
// they’re not already enabled by the language standard.

// Driver warnings.
// yes-warning@* {{ignoring '-fraw-string-literals', which is only valid for C}}
// no-warning@* {{ignoring '-fno-raw-string-literals', which is only valid for C}}

void f() {
  (void) R"foo()foo"; // unsupported-error {{use of undeclared identifier 'R'}} cxx-unsupported-error {{expected ';' after expression}}
  (void) LR"foo()foo"; // unsupported-error {{use of undeclared identifier 'LR'}} cxx-unsupported-error {{expected ';' after expression}}
  (void) uR"foo()foo"; // unsupported-error {{use of undeclared identifier 'uR'}} cxx-unsupported-error {{expected ';' after expression}}
  (void) u8R"foo()foo"; // unsupported-error {{use of undeclared identifier 'u8R'}} cxx-unsupported-error {{expected ';' after expression}}
  (void) UR"foo()foo"; // unsupported-error {{use of undeclared identifier 'UR'}} cxx-unsupported-error {{expected ';' after expression}}
}

// supported-error@* {{missing terminating delimiter}}
// supported-error@* {{expected expression}}
// supported-error@* {{expected ';' after top level declarator}}
#define R "bar"
const char* s =  R"foo(";
