// RUN: %clang_cc1 -verify=c2x -std=c2x %s
// RUN: %clang_cc1 -verify=c11 -std=c11 %s
// RUN: %clang_cc1 -verify=gnu11 -std=gnu11 %s
// RUN: %clang_cc1 -verify=pedantic -pedantic -std=gnu11 -Wno-comment %s
// RUN: %clang_cc1 -verify=compat -std=c2x -Wpre-c2x-compat %s

// c2x-no-diagnostics

// Exercise the various circumstances under which we will diagnose use of
// typeof and typeof_unqual as either an extension or as a compatability
// warning. Note that GCC exposes 'typeof' as a non-conforming extension in
// standards before C2x, and Clang has followed suit. Neither compiler exposes
// 'typeof_unqual' as a non-conforming extension.

// Show what happens with the underscored version of the keyword, which is a
// conforming extension.
__typeof__(int) i = 12;

// Show what happens with a regular 'typeof' use.
typeof(i) j = 12; // c11-error {{expected function body after function declarator}} \
                     pedantic-warning {{extension used}} \
                     compat-warning {{'typeof' is incompatible with C standards before C2x}}

// Same for 'typeof_unqual'.
typeof_unqual(j) k = 12; // c11-error {{expected function body after function declarator}} \
                            gnu11-error {{expected function body after function declarator}} \
                            pedantic-error {{expected function body after function declarator}} \
                            compat-warning {{'typeof_unqual' is incompatible with C standards before C2x}}

