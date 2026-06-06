// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 %t/parta.cppm -o %t/mod-parta.pcm -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/impl.cppm -o %t/mod-impl.pcm -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/primary.cppm -o %t/mod.pcm -fsyntax-only -verify

//--- parta.cppm
export module mod:parta;

module :private; // expected-error {{private module fragment declaration with no preceding module declaration}}

//--- impl.cppm

module mod:impl;

module :private; // expected-error {{private module fragment declaration with no preceding module declaration}}

//--- primary.cppm
//expected-no-diagnostics
export module mod;

module :private;

