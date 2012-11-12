// RUN: %clang -fsyntax-only -target i686-pc-win32 %s
// RUN: %clang -fsyntax-only -target i386-mingw32 %s

// Something in MSVC's headers (pulled in e.g. by <crtdefs.h>) defines __null
// to something, mimick that.
#define __null

#include <stddef.h>

// __null is used as a type annotation in MS headers, with __null defined to
// nothing in regular builds. This should continue to work even with stddef.h
// included.
void f(__null void* p) { }

// NULL should work fine even with __null defined to nothing.
void* p = NULL;
