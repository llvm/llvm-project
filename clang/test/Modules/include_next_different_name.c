// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -I%t/x -I%t/y -fsyntax-only -verify %t/main.c
// RUN: %clang_cc1 -I%t/x -I%t/y -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -fsyntax-only -verify %t/main.c
// RUN: %clang_cc1 -I%t/x -I%t/y -fsyntax-only -verify %t/main_sub.c
// RUN: %clang_cc1 -I%t/x -I%t/y -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -fsyntax-only -verify %t/main_sub.c

// A header in a module doing '#include_next' of a header with a different name
// must continue the search after the directory the current header was found in,
// not restart from the first include directory.

//--- x/wrapper.h
#ifndef X_WRAPPER_H
#define X_WRAPPER_H
int wrapper_x_marker(void) { return 1; }
#endif

//--- y/wrapper.h
#ifndef Y_WRAPPER_H
#define Y_WRAPPER_H
int wrapper_y_marker(void) { return 2; }
#endif

//--- x/inner.h
#ifndef X_INNER_H
#define X_INNER_H
// Found in 'x', so the search must continue in 'y' and find y/wrapper.h.
#include_next <wrapper.h>
#endif

//--- x/module.modulemap
module Inner { header "inner.h" export * }

//--- main.c
// expected-no-diagnostics
#include <inner.h>
int use(void) { return wrapper_y_marker(); }

// The current header lives in a subdirectory of the include directory
// ("#include <sys/inner.h>"); that subdirectory must be considered when
// locating the directory the header was found in.

//--- x/sys/inner.h
#ifndef X_SYS_INNER_H
#define X_SYS_INNER_H
#include_next <wrapper.h>
#endif

//--- x/sys/module.modulemap
module SysInner { header "inner.h" export * }

//--- main_sub.c
// expected-no-diagnostics
#include <sys/inner.h>
int use_sub(void) { return wrapper_y_marker(); }
