// RUN: %clang_cc1 %s -triple arm64-apple-macosx -fsyntax-only -verify

register struct Undefined1 bar1 asm("x1"); // #inline-type-def
// expected-error@#inline-type-def {{tentative definition has type 'struct Undefined1' that is never completed}}
// expected-note@#inline-type-def {{forward declaration of 'struct Undefined1'}}
struct Undefined2; // #outline-type-def
register struct Undefined2 bar2 asm("x1"); // #outline-type-label
// expected-error@#outline-type-label {{tentative definition has type 'struct Undefined2' that is never completed}}
// expected-note@#outline-type-def {{forward declaration of 'struct Undefined2'}}
