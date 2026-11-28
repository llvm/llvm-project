// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -I%t/inc -I%t/inc/v1 -I%t/sys -fsyntax-only -verify %t/main.c
// RUN: %clang_cc1 -I%t/inc -I%t/inc/v1 -I%t/sys -fmodules \
// RUN:   -fimplicit-module-maps -fmodules-cache-path=%t/cache -fsyntax-only \
// RUN:   -verify %t/main.c

// One search directory is nested in another (like libc++'s <include> and
// <include/c++/v1>). Wrapper headers in the inner directory doing
// '#include_next' must continue after the inner directory, i.e. the header is
// located under the name relative to the innermost search directory.

//--- inc/v1/header.h
#ifndef V1_HEADER_H
#define V1_HEADER_H
#include_next <header.h>
#endif

//--- inc/v1/other.h
#ifndef V1_OTHER_H
#define V1_OTHER_H
#include_next <sysother.h>
#endif

// Same name as the system header but must not be picked.
//--- inc/v1/sysother.h
#ifndef V1_SYSOTHER_H
#define V1_SYSOTHER_H
#endif

//--- inc/v1/module.modulemap
module Header { header "header.h" export * }
module Other { header "other.h" export * }

//--- sys/header.h
#ifndef SYS_HEADER_H
#define SYS_HEADER_H
int sys_header_marker(void);
#endif

//--- sys/sysother.h
#ifndef SYS_SYSOTHER_H
#define SYS_SYSOTHER_H
int sys_other_marker(void);
#endif

//--- main.c
// expected-no-diagnostics
#include <header.h>
#include <other.h>
int use(void) { return sys_header_marker() + sys_other_marker(); }
