// RUN: %clang_cc1 -std=c++2a -DEXPORT %s -verify
// RUN: %clang_cc1 -std=c++2a -DEXPORT %s -emit-module-interface -o %t.pcm
// RUN: %clang_cc1 -std=c++2a -UEXPORT %s -verify -fmodule-file=%t.pcm

#ifdef EXPORT
// expected-no-diagnostics
export
#else
// expected-note@+2 {{add 'export' here}}
#endif
module M;

#ifndef EXPORT
// expected-error@+2 {{private module fragment in module implementation unit}}
#endif
module :private;
