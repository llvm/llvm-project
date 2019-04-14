// RUN: %clang_cc1 -std=c++2a -verify %s

export module foo:bar; // expected-error {{sorry, module partitions are not yet supported}}
import :baz; // expected-error {{sorry, module partitions are not yet supported}}
