// RUN: %clang_cc1 -fsyntax-only -std=c++1z -verify %s

struct [[nodiscard]] S1 {}; // ok
struct [[nodiscard nodiscard]] S2 {}; // expected-error {{attribute 'nodiscard' cannot appear multiple times in an attribute specifier}}
struct [[nodiscard("Wrong")]] S3 {}; // expected-error {{'nodiscard' cannot have an argument list}}

[[nodiscard]] int f();
enum [[nodiscard]] E {};

namespace [[nodiscard]] N {} // expected-warning {{'nodiscard' attribute only applies to functions, methods, enums, and classes}}
