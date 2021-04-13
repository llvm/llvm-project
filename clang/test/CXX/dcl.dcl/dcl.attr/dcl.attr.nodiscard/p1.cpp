// RUN: %clang_cc1 -fsyntax-only -std=c++2a -verify %s

struct [[nodiscard]] S1 {}; // ok
struct [[nodiscard, nodiscard]] S2 {}; // ok
struct [[nodiscard("Wrong")]] S3 {};

[[nodiscard]] int f();
enum [[nodiscard]] E {};

namespace [[nodiscard]] N {} // expected-warning {{'nodiscard' attribute only applies to Objective-C methods, enums, structs, unions, classes, functions, and function pointers}}
