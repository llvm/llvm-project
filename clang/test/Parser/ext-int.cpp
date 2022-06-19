// RUN: %clang_cc1 -fsyntax-only -verify %s

// expected-error@+5{{expected ')'}}
// expected-note@+4{{to match this '('}}
// expected-error@+3{{expected unqualified-id}}
// expected-error@+2{{extraneous closing brace}}
// expected-error@+1{{a type specifier is required for all declarations}}
_BitInt(32} a;
// expected-error@+2{{expected expression}}
// expected-error@+1{{a type specifier is required for all declarations}}
_BitInt(32* ) b;
// expected-error@+3{{expected '('}}
// expected-error@+2{{expected unqualified-id}}
// expected-error@+1{{a type specifier is required for all declarations}}
_BitInt{32} c;
