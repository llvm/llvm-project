// RUN: rm -rf %t
// RUN: mkdir %t

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   -F%S/Inputs/at-import-in-framework-header -I%S/Inputs/at-import-in-framework-header \
// RUN:   -Watimport-in-framework-header -fsyntax-only %s \
// RUN:   2>%t/stderr
// RUN: FileCheck --input-file=%t/stderr %s

// CHECK: use of '@import' in framework header is discouraged

#import <A/A.h>

int bar() { return foo(); }

