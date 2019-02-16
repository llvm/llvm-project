// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps -I%S/Inputs/import-textual -fmodules-cache-path=%t %s -verify

// expected-no-diagnostics

#import "x.h"
#import "x.h"

