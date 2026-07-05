// RUN: %clang_cc1 -verify -ast-print %s -xc -DDEF_BOOL_CBOOL \
// RUN: | FileCheck %s --check-prefixes=BOOL-AS-CBOOL,CBOOL
//
// RUN: %clang_cc1 -verify -ast-print %s -xc -DDEF_BOOL_CBOOL -DDIAG \
// RUN: | FileCheck %s --check-prefixes=BOOL-AS-CBOOL,CBOOL
//
// RUN: %clang_cc1 -verify -ast-print %s -xc -DDEF_BOOL_INT \
// RUN: | FileCheck %s --check-prefixes=BOOL-AS-INT,CBOOL
//
// RUN: %clang_cc1 -verify -ast-print %s -xc -DDEF_BOOL_INT -DDIAG \
// RUN: | FileCheck %s --check-prefixes=BOOL-AS-INT,CBOOL
//
// RUN: %clang_cc1 -verify -ast-print %s -xc++ \
// RUN: | FileCheck %s --check-prefixes=BOOL-AS-BOOL
//
// RUN: %clang_cc1 -verify -ast-print %s -xc++ -DDIAG \
// RUN: | FileCheck %s --check-prefixes=BOOL-AS-BOOL

#if DEF_BOOL_CBOOL
# define bool _Bool
#elif DEF_BOOL_INT
# define bool int
#endif

// BOOL-AS-CBOOL: _Bool i;
// BOOL-AS-INT:   int i;
// BOOL-AS-BOOL:  bool i;
bool i;

#ifndef __cplusplus
// CBOOL: _Bool j;
_Bool j;
#endif

// Induce a diagnostic (and verify we actually managed to do so), which used to
// permanently alter the -ast-print printing policy for _Bool.  How bool is
// defined by the preprocessor is examined only once per compilation, when the
// diagnostic is emitted, and it used to affect the entirety of -ast-print, so
// test only one definition of bool per compilation.
#if DIAG
void fn(void) { 1; } // expected-warning {{expression result unused}}
#else
// expected-no-diagnostics
#endif
