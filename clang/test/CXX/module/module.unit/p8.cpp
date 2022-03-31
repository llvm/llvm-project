// RUN: echo 'export module foo; export int n;' > %t.cppm
// RUN: %clang_cc1 -std=c++2a %t.cppm -emit-module-interface -o %t.pcm
// RUN: %clang_cc1 -std=c++2a -fmodule-file=%t.pcm -verify -DMODE=0 %s
// RUN: %clang_cc1 -std=c++2a -fmodule-file=%t.pcm -verify -DMODE=1 %s
// RUN: %clang_cc1 -std=c++2a -fmodule-file=%t.pcm -verify -DMODE=2 %s
// RUN: %clang_cc1 -std=c++2a -fmodule-file=%t.pcm -verify -DMODE=3 %s
// RUN: %clang_cc1 -std=c++2a -fmodule-file=%t.pcm -verify -DMODE=4 %s
// RUN: %clang_cc1 -std=c++2a -fmodule-file=%t.pcm -verify -DMODE=5 %s

#if MODE == 0
// no module declaration

#elif MODE == 1
// expected-no-diagnostics
module foo; // Implementation, implicitly imports foo.
#define IMPORTED

#elif MODE == 2
export module foo; // expected-error {{redefinition of module 'foo'}}
// expected-note-re@* {{module loaded from '{{.*}}.pcm'}}
#define IMPORTED

#elif MODE == 3
export module bar; // A different module

#elif MODE == 4
module foo:bar; // Partition implementation
//#define IMPORTED (we don't import foo here)

#elif MODE == 5
export module foo:bar; // Partition interface
//#define IMPORTED  (we don't import foo here)

#endif

int k = n;
#ifndef IMPORTED
// expected-error@-2 {{declaration of 'n' must be imported from module 'foo' before it is required}}
// expected-note@* {{not visible}}
#endif
