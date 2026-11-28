// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -I%t/inc -I%t/inc/v1 -I%t/sys -fsyntax-only -verify %t/main.c
// RUN: %clang_cc1 -I%t/inc -I%t/inc/v1 -I%t/sys -fmodules \
// RUN:   -fimplicit-module-maps -fmodules-cache-path=%t/cache -fsyntax-only \
// RUN:   -verify %t/main.c

// One search directory is nested in another, and the header is found through
// the outer one (as "v1/other.h"). The '#include_next' must continue after the
// outer directory, so it finds the header in the inner directory, rather than
// assuming the header was found through the inner one and skipping to 'sys'.

//--- inc/v1/other.h
#ifndef V1_OTHER_H
#define V1_OTHER_H
#include_next <target.h>
#endif

//--- inc/v1/target.h
#ifndef V1_TARGET_H
#define V1_TARGET_H
int v1_target_marker(void);
#endif

//--- inc/module.modulemap
module Other { header "v1/other.h" export * }

//--- sys/target.h
#ifndef SYS_TARGET_H
#define SYS_TARGET_H
int sys_target_marker(void);
#endif

//--- main.c
// expected-no-diagnostics
#include <v1/other.h>
int use(void) { return v1_target_marker(); }
