// RUN: %clang -cc1 -E -fmodules %s -verify

// Just checking the syntax here; the semantics are tested elsewhere.
#pragma clang module import // expected-error {{expected identifier in module name}}
#pragma clang module import ! // expected-error {{expected identifier in module name}}
#pragma clang module import if // expected-error {{expected identifier in module name}}
#pragma clang module import foo ? bar // expected-error {{expected '.' or end of directive after module name}}
#pragma clang module import foo. // expected-error {{expected identifier}}
#pragma clang module import foo.bar.baz.quux // expected-error {{module 'foo' not found}}

#error here // expected-error {{here}}
