// RUN: cp %s %t
// RUN: %clang_cc1 -pedantic -fixit %t
// RUN: %clang_cc1 -pedantic -Werror -x c %t

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. All of the
   warnings will be fixed by -fixit, and the resulting file should
   compile cleanly with -Werror -pedantic. */

// FIXME: If you put a space at the end of the line, it doesn't work yet!
char *s = "hi\
there";

// The following line isn't terminated, don't fix it.
int i; // expected-error{{no newline at end of file}}
