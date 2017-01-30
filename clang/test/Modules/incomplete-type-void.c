// RUN: rm -rf %t.cache
// RUN: %clang_cc1 -fmodules %s -fmodules-cache-path=%t.cache \
// RUN:     -fsyntax-only -I %S/Inputs/incomplete-type -verify

// expected-no-diagnostics

#import "C.h"
#import "A.h"
void foo __P(());
