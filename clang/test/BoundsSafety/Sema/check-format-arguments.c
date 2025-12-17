
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify -Wformat-nonliteral %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify -Wformat-nonliteral %s

#include <ptrcheck.h>
#include <stdarg.h>

#define __printflike(__fmt,__varargs) __attribute__((__format__ (__printf__, __fmt, __varargs)))

void __printflike(1, 0) foo(const char *__null_terminated, va_list);

// expected-note@+1{{'bar' declared here}}
void __printflike(2, 3) bar(const char *__unsafe_indexable p1, const char *__unsafe_indexable p2, ...) {
    va_list variadicArgs;
    va_start(variadicArgs, p2);

    foo(__unsafe_forge_null_terminated(const char *, p2), variadicArgs);
    foo(__unsafe_forge_null_terminated(const char *, p2+1), variadicArgs);
    foo(__unsafe_forge_null_terminated(const char *, "Hello, %s!\n"), variadicArgs);

    foo(__unsafe_forge_null_terminated(const char *, 2), variadicArgs); // expected-warning{{format string is not a string literal}}
    foo(__unsafe_forge_null_terminated(const char *, p1), variadicArgs); // expected-warning{{diagnostic behavior may be improved by adding the 'format(printf, 1, 3)' attribute to the declaration of 'bar'}}

    va_end(variadicArgs);
}
