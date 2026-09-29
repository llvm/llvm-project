// RUN: %clang_cc1 -fsyntax-only -std=gnu11 -verify=supported %s
// RUN: %clang_cc1 -fsyntax-only -std=c11 -DUNICODE -fraw-string-literals -verify=supported %s
// RUN: %clang_cc1 -fsyntax-only -std=gnu89 -verify=unsupported %s
// RUN: %clang_cc1 -fsyntax-only -std=c11 -DUNICODE -verify=unsupported %s
// RUN: %clang_cc1 -fsyntax-only -std=gnu11 -DUNICODE -fno-raw-string-literals -verify=unsupported %s

// RUN: %clang_cc1 -x c++ -fsyntax-only -Wno-unused -std=c++03 -verify=unsupported,cxx-unsupported %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wno-unused -std=gnu++03 -verify=unsupported,cxx-unsupported %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wno-unused -std=c++03 -fraw-string-literals -verify=supported %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wno-unused -std=gnu++03 -fraw-string-literals -verify=supported %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wno-unused -std=c++11 -DUNICODE -verify=supported,cxx %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wno-unused -std=gnu++11 -DUNICODE -verify=supported,cxx %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wno-unused -std=c++11 -DUNICODE -fraw-string-literals -verify=supported,yes %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wno-unused -std=gnu++11 -DUNICODE -fraw-string-literals -verify=supported,yes %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wno-unused -std=c++11 -DUNICODE -fno-raw-string-literals -verify=supported,no %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wno-unused -std=gnu++11 -DUNICODE -fno-raw-string-literals -verify=supported,no %s

// GCC supports raw string literals in C99 and later in '-std=gnuXY' mode; we
// additionally provide '-f[no-]raw-string-literals' to enable/disable them
// explicitly in C.
//
// We do not allow disabling raw string literals in C++ mode if they’re enabled
// by the language standard, i.e. in C++11 or later.

// Driver warnings.
// yes-warning@* {{ignoring '-fraw-string-literals'}}
// no-warning@* {{ignoring '-fno-raw-string-literals'}}

void f() {
  (void) R"foo()foo"; // unsupported-error {{use of undeclared identifier 'R'}}
  (void) LR"foo()foo"; // unsupported-error {{use of undeclared identifier 'LR'}}

#ifdef UNICODE
  (void) uR"foo()foo"; // unsupported-error {{use of undeclared identifier 'uR'}}
  (void) u8R"foo()foo"; // unsupported-error {{use of undeclared identifier 'u8R'}}
  (void) UR"foo()foo"; // unsupported-error {{use of undeclared identifier 'UR'}}
#endif
}

// supported-error@* {{missing terminating delimiter}}
// supported-error@* {{expected expression}}
// supported-error@* {{expected ';' after top level declarator}}
#define R "bar"
const char* s =  R"foo(";
