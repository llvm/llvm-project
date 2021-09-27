// RUN: %clang_cc1 -Wdeprecated %s -fsyntax-only -verify

// expected-error@+1{{expected (}}
#pragma clang deprecated

// expected-error@+1{{expected identifier}}
#pragma clang deprecated(4

// expected-error@+1{{no macro named 'foo'}}
#pragma clang deprecated(foo)

#define bar 1
// expected-note@+1{{macro marked 'deprecated' here}} 
#pragma clang deprecated(bar, "bar is deprecated use 1")

// expected-warning@+1{{macro 'bar' has been marked as deprecated: bar is deprecated use 1}}
#if bar
#endif

#define foo 1
// expected-note@+8{{macro marked 'deprecated' here}} 
// expected-note@+7{{macro marked 'deprecated' here}} 
// expected-note@+6{{macro marked 'deprecated' here}} 
// expected-note@+5{{macro marked 'deprecated' here}} 
// expected-note@+4{{macro marked 'deprecated' here}} 
// expected-note@+3{{macro marked 'deprecated' here}} 
// expected-note@+2{{macro marked 'deprecated' here}} 
// expected-note@+1{{macro marked 'deprecated' here}} 
#pragma clang deprecated(foo)

// expected-error@+1{{expected )}}
#pragma clang deprecated(foo

// expected-warning@+1{{macro 'foo' has been marked as deprecated}}
#if foo
#endif

// expected-warning@+1{{macro 'foo' has been marked as deprecated}}
#if defined(foo)
#endif

// expected-warning@+1{{macro 'foo' has been marked as deprecated}}
#ifdef foo
#endif

// expected-warning@+1{{macro 'foo' has been marked as deprecated}}
#ifndef foo
#endif

int main(int argc, char** argv) {
// expected-error@+1{{no macro named 'main'}}
#pragma clang deprecated(main)

  // expected-warning@+1{{macro 'foo' has been marked as deprecated}}
  return foo;
}

#define frobble 1
#pragma clang deprecated(frobble)

// not-expected-warning@+1{{macro 'frobble' has been marked as deprecated}}
#undef frobble // Expect no diagnostics here

// not-expected-warning@+1{{macro 'frobble' has been marked as deprecated}}
#define frobble 1 // How about here given that this was undefined?

// not-expected-warning@+1{{macro 'frobble' has been marked as deprecated}}
#if defined(frobble)
#endif

// Test that we diagnose on #elif.
#if 0
#elif foo
// expected-warning@-1{{macro 'foo' has been marked as deprecated}}
#endif


// Test that we diagnose on #elifdef.
#ifdef baz
#elifdef foo
// expected-warning@-1{{macro 'foo' has been marked as deprecated}}
#endif

// Test that we diagnose on #elifndef.
#ifdef baz
#elifndef foo
#endif
// expected-warning@-2{{macro 'foo' has been marked as deprecated}}

// FIXME: These cases are currently not handled because clang doesn't expand
// conditions on skipped #elif* blocks. See the FIXME notes in
// Preprocessor::SkipExcludedConditionalBlock.

#ifdef frobble
// not-expected-warning@+1{{macro 'foo' has been marked as deprecated}}
#elifndef foo
#endif

#ifdef frobble
// not-expected-warning@+1{{macro 'foo' has been marked as deprecated}}
#elifdef foo
#endif

#if 1
// not-expected-warning@+1{{macro 'foo' has been marked as deprecated}}
#elif foo
#endif
