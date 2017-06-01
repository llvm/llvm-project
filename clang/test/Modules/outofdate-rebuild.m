// RUN: rm -rf %t.cache
// RUN: echo "@import CoreText;" > %t.m
// RUN: %clang_cc1 -fdisable-module-hash -fmodules-cache-path=%t.cache \
// RUN:   -fmodules -fimplicit-module-maps -I%S/Inputs/outofdate-rebuild %s \
// RUN:   -fsyntax-only
// RUN: %clang_cc1 -DMISMATCH -Werror -fdisable-module-hash \
// RUN:   -fmodules-cache-path=%t.cache -fmodules -fimplicit-module-maps \
// RUN:   -I%S/Inputs/outofdate-rebuild %t.m -fsyntax-only
// RUN: %clang_cc1 -fdisable-module-hash -fmodules-cache-path=%t.cache \
// RUN:   -fmodules -fimplicit-module-maps -I%S/Inputs/outofdate-rebuild %s \
// RUN:   -fsyntax-only

// This testcase reproduces a use-after-free in when ModuleManager removes an
// entry from the PCMCache without notifying its parent ASTReader.
@import Cocoa;
