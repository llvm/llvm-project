// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
// RUN: echo 'export module foo;' > %t.cppm
// RUN: echo 'export int n;' >> %t.cppm
// RUN: %clang_cc1 -std=c++2a %t.cppm -emit-module-interface -o %t.pcm
// RUN: %clang_cc1 -std=c++2a -fmodule-file=foo=%t.pcm -verify -DMODE=0 %t/A.cppm
// RUN: %clang_cc1 -std=c++2a -fmodule-file=foo=%t.pcm -verify -DMODE=1 %t/B.cppm
// RUN: %clang_cc1 -std=c++2a -fmodule-file=foo=%t.pcm -verify -DMODE=2 %t/C.cppm
// RUN: %clang_cc1 -std=c++2a -fmodule-file=foo=%t.pcm -verify -DMODE=3 %t/D.cppm
// RUN: %clang_cc1 -std=c++2a -fmodule-file=foo=%t.pcm -verify -DMODE=4 %t/E.cppm
// RUN: %clang_cc1 -std=c++2a -fmodule-file=foo=%t.pcm -verify -DMODE=5 %t/F.cppm

//--- A.cppm
// no module declaration
// expected-no-diagnostics

//--- B.cppm
// expected-no-diagnostics
module foo; // Implementation, implicitly imports foo.
#define IMPORTED

int k = n;

//--- C.cppm
export module foo;

int k = n; // expected-error {{use of undeclared identifier 'n'}}

//--- D.cppm
export module bar; // A different module

int k = n; // expected-error {{use of undeclared identifier 'n'}}

//--- E.cppm
module foo:bar; // Partition implementation
//#define IMPORTED (we don't import foo here)

int k = n; // expected-error {{use of undeclared identifier 'n'}}

//--- F.cppm
export module foo:bar; // Partition interface
//#define IMPORTED  (we don't import foo here)

int k = n; // expected-error {{use of undeclared identifier 'n'}}
