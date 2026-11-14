// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t && %clang_cc1 -I%t/z -I%t/x -I%t/y -fmodules \
// RUN:   -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   -Werror=shadow-header -fsyntax-only main.c

// The current header is reached through a path that is not textually prefixed
// by its search directory ("./x/sys/inner.h" for -I%t/x), and an unrelated
// header with the same relative name comes first in the search path. Locating
// the current header's directory must still work, and must not report
// -Wshadow-header for the headers it probes.

//--- z/sys/inner.h
int unrelated_z_marker(void) { return 3; }

//--- x/sys/inner.h
#ifndef X_SYS_INNER_H
#define X_SYS_INNER_H
#include_next <wrapper.h>
#endif

//--- x/sys/module.modulemap
module SysInner { header "inner.h" export * }

//--- x/wrapper.h
int wrapper_x_marker(void) { return 1; }

//--- y/wrapper.h
int wrapper_y_marker(void) { return 2; }

//--- main.c
#include "x/sys/inner.h"
int use(void) { return wrapper_y_marker(); }
