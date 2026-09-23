// RUN: %clang_cc1 -fms-compatibility -Wno-microsoft-enum-forward-reference -fsyntax-only -verify %s
// expected-no-diagnostics

typedef enum Original { Value } Alias;
enum Alias *tag_pointer;
Alias *typedef_pointer;

_Static_assert(!__builtin_types_compatible_p(enum Alias, Alias),
               "the enum tag and typedef must remain distinct in C");
