// RUN: %clang_cc1 %s -fsyntax-only -pedantic -verify -std=c99
// RUN: %clang_cc1 %s -fsyntax-only -pedantic -verify -DPREDECLARE -std=c99

#ifdef PREDECLARE
// PR16344
// Clang has defined 'vfprint' in builtin list. If the following line occurs before any other
// `vfprintf' in this file, and we getPreviousDecl()->getTypeSourceInfo() on it, then we will
// get a null pointer since the one in builtin list doesn't has valid TypeSourceInfo.
int vfprintf(void) { return 0; } // expected-warning {{requires inclusion of the header <stdio.h>}}
#endif

// PR4290
// The following declaration is compatible with vfprintf, so we shouldn't
// reject.
int vfprintf(); // expected-warning {{a function declaration without a prototype is deprecated in all versions of C}}
#ifndef PREDECLARE
// expected-warning@-2 {{requires inclusion of the header <stdio.h>}}
#endif
