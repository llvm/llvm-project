// RUN: %clang_cc1 --std=c++23 -fsyntax-only -verify %s
// see ISSUE 219950 and PR 223044
module;
module :private; // expected-error {{private module fragment declaration with no preceding module declaration}}
export module Foo;
