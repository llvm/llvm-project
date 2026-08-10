// RUN: %clang_cc1 %s -verify -fsyntax-only -Wnewline-eof
// vim: set binary noeol:

// This file intentionally ends without a \n on the last line.  Make sure your
// editor doesn't add one.

// expected-error@+2{{unterminated conditional directive}}
// expected-warning@+1{{no newline at end of file}}
#if 0