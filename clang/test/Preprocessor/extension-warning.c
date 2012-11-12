// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s

// The preprocessor shouldn't warn about extensions within macro bodies that
// aren't expanded.
#define TY typeof
#define TY1 typeof(1)

// But we should warn here
TY1 x; // expected-warning {{extension}}
TY(1) x; // FIXME: And we should warn here

// Note: this warning intentionally doesn't trigger on keywords like
// __attribute; the standard allows implementation-defined extensions
// prefixed with "__".
// Current list of keywords this can trigger on:
// inline, restrict, asm, typeof, _asm

void whatever() {}
