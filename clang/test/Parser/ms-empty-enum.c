// RUN: %clang_cc1 %s -fsyntax-only -Wmicrosoft -verify -fms-extensions
// RUN: %clang_cc1 %s -fsyntax-only -Wmicrosoft -verify -fms-compatibility
// expected-no-diagnostics

typedef enum tag1 { } A;
typedef enum tag2 { } B;
typedef enum : unsigned { } C;

