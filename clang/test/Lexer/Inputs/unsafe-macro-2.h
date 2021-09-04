// expected-warning@+1{{macro 'UNSAFE_MACRO' has been marked as unsafe for use in headers: Don't use this!}}
#if UNSAFE_MACRO
#endif

// expected-warning@+1{{macro 'UNSAFE_MACRO' has been marked as unsafe for use in headers: Don't use this!}}
#if defined(UNSAFE_MACRO)
#endif

// expected-warning@+1{{macro 'UNSAFE_MACRO' has been marked as unsafe for use in headers: Don't use this!}}
#ifdef UNSAFE_MACRO
#endif

// expected-warning@+1{{macro 'UNSAFE_MACRO' has been marked as unsafe for use in headers: Don't use this!}}
#ifndef UNSAFE_MACRO
#endif

// expected-warning@+1{{macro 'UNSAFE_MACRO' has been marked as unsafe for use in headers: Don't use this!}}
const int x = UNSAFE_MACRO;

// expected-warning@+1{{macro 'UNSAFE_MACRO_2' has been marked as unsafe for use in headers}}
const int y = UNSAFE_MACRO_2;

// not-expected-warning@+1{{macro 'UNSAFE_MACRO_2' has been marked as unsafe for use in headers}}
#undef UNSAFE_MACRO_2
// not-expected-warning@+1{{macro 'UNSAFE_MACRO_2' has been marked as unsafe for use in headers}}
#define UNSAFE_MACRO_2 2

// not-expected-warning@+1{{macro 'UNSAFE_MACRO_2' has been marked as unsafe for use in headers}}
const int z = UNSAFE_MACRO_2;


// Test that we diagnose on #elif.
#if 0
#elif UNSAFE_MACRO
// expected-warning@-1{{macro 'UNSAFE_MACRO' has been marked as unsafe for use in headers: Don't use this!}}
#endif


// Test that we diagnose on #elifdef.
#ifdef baz
#elifdef UNSAFE_MACRO
// expected-warning@-1{{macro 'UNSAFE_MACRO' has been marked as unsafe for use in headers: Don't use this!}}
#endif

// Test that we diagnose on #elifndef.
#ifdef baz
#elifndef UNSAFE_MACRO
#endif
// expected-warning@-2{{macro 'UNSAFE_MACRO' has been marked as unsafe for use in headers: Don't use this!}}

// FIXME: These cases are currently not handled because clang doesn't expand
// conditions on skipped #elif* blocks. See the FIXME notes in
// Preprocessor::SkipExcludedConditionalBlock.

#define frobble

#ifdef frobble
// not-expected-warning@+1{{macro 'UNSAFE_MACRO' has been marked as unsafe for use in headers: Don't use this!}}
#elifndef UNSAFE_MACRO
#endif

#ifdef frobble
// not-expected-warning@+1{{macro 'UNSAFE_MACRO' has been marked as unsafe for use in headers: Don't use this!}}
#elifdef UNSAFE_MACRO
#endif

#if 1
// not-expected-warning@+1{{macro 'UNSAFE_MACRO' has been marked as unsafe for use in headers: Don't use this!}}
#elif UNSAFE_MACRO
#endif
