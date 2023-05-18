// RUN: %clang_cc1 -E %s | FileCheck -strict-whitespace %s

#define R_PAREN ) 

#define FUNC(a) a 

static int glob = (1 + FUNC(1 R_PAREN ); 

// CHECK: static int glob = (1 + 1 );

