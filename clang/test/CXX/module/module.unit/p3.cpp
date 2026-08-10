// RUN: %clang_cc1 -std=c++2a -verify %s

export module foo:bar;
import :baz; // expected-error {{module 'foo:baz' not found}}
