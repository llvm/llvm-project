// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++2a %t/pmf_in_interface.cpp -verify
// RUN: %clang_cc1 -std=c++2a %t/pmf_in_interface.cpp -emit-module-interface -o %t.pcm
// RUN: %clang_cc1 -std=c++2a %t/pmf_in_implementation.cpp -verify -fmodule-file=M=%t.pcm


//--- pmf_in_interface.cpp
// expected-no-diagnostics
export module M;
module :private;

//--- pmf_in_implementation.cpp
module M; // expected-note {{add 'export' here}}
module :private; // expected-error {{private module fragment in module implementation unit}}
