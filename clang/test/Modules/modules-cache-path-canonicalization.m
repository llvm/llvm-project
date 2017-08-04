// RUN: rm -rf %t/cache %T/rel

// This testcase reproduces a use-after-free after looking up a PCM in
// a non-canonical modules-cache-path.
//
// Prime the module cache (note the '.' in the path).
// RUN: %clang_cc1 -fdisable-module-hash -fmodules-cache-path=%t/./cache \
// RUN:   -fmodules -fimplicit-module-maps -I %S/Inputs/outofdate-rebuild \
// RUN:   %s -fsyntax-only
//
// Force a module to be rebuilt by creating a conflict.
// RUN: echo "@import CoreText;" > %t.m
// RUN: %clang_cc1 -DMISMATCH -Werror -fdisable-module-hash \
// RUN:   -fmodules-cache-path=%t/./cache -fmodules -fimplicit-module-maps \
// RUN:   -I %S/Inputs/outofdate-rebuild %t.m -fsyntax-only
//
// Rebuild.
// RUN: %clang_cc1 -fdisable-module-hash -fmodules-cache-path=%t/./cache \
// RUN:   -fmodules -fimplicit-module-maps -I %S/Inputs/outofdate-rebuild \
// RUN:   %s -fsyntax-only


// Unrelated to the above: Check that a relative path is resolved correctly.
//
// RUN: %clang_cc1 -working-directory %T/rel -fmodules-cache-path=./cache \
// RUN:   -fmodules -fimplicit-module-maps -I %S/Inputs/outofdate-rebuild \
// RUN:   -fdisable-module-hash %t.m -fsyntax-only -Rmodule-build 2>&1 \
// RUN:   | FileCheck %s
// CHECK: {{/|\\}}rel{{/|\\}}cache{{/|\\}}CoreText.pcm
@import Cocoa;
