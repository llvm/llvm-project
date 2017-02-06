// RUN: rm -rf %t.cache
// RUN: echo "@import CoreText;" > %t.m
// RUN: %clang_cc1 -fdisable-module-hash -fmodules-cache-path=%t.cache -fmodules -fimplicit-module-maps -I %S/Inputs/outofdate-rebuild %s -fsyntax-only -Rmodule-build
// RUN: echo -----------------------------------------------
// RUN: %clang_cc1 -DMISMATCH -Werror -fdisable-module-hash -fmodules-cache-path=%t.cache -fmodules -fimplicit-module-maps -I %S/Inputs/outofdate-rebuild %t.m -fsyntax-only -Rmodule-build
// RUN: echo -----------------------------------------------
// RUN: %clang_cc1 -fdisable-module-hash -fmodules-cache-path=%t.cache -fmodules -fimplicit-module-maps -I %S/Inputs/outofdate-rebuild %s -fsyntax-only -Rmodule-build

// This testcase reproduces a use-after-free in
// https://reviews.llvm.org/D28299 when ModuleManager removes an entry
// from the PCMcache without notifying its parent ASTReader.
@import Cocoa;
