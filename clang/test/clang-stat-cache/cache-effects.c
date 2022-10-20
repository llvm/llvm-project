#include "foo.h"

// Testing the effects of a cache is tricky, because it's just supposed to speed
// things up, not change the behavior. In this test, we are using an outdated
// cache to trick HeaderSearch into finding the wrong module and show that it is
// being used.

// Clear the module cache.
// RUN: rm -rf %t
// RUN: mkdir -p %t/Inputs
// RUN: mkdir -p %t/Inputs/Foo1
// RUN: mkdir -p %t/Inputs/Foo2
// RUN: mkdir -p %t/modules-to-compare

// ===
// Create a Foo module in the Foo1 direcotry.
// RUN: echo 'void meow(void);' > %t/Inputs/Foo1/foo.h
// RUN: echo 'module Foo { header "foo.h" }' > %t/Inputs/Foo1/module.map

// ===
// Compile the module. Note that the compiler has 2 header search paths:
// Foo2 and Foo1 in that order. The module has been created in Foo1, and
// it is the only version available now.
// RUN: %clang_cc1 -cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -fsyntax-only -I %t/Inputs/Foo2 -I %t/Inputs/Foo1 -Rmodule-build  %s 2>&1
// RUN: cp %t/modules-cache/Foo.pcm %t/modules-to-compare/Foo-before.pcm

// ===
// Create a stat cache for our inputs directory
// RUN: clang-stat-cache %t/Inputs -o %t/stat.cache

// ===
// As a sanity check, re-run the same compilation with the cache and check that
// the module does not change.
// RUN: %clang_cc1 -cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -fsyntax-only -I %t/Inputs/Foo2 -I %t/Inputs/Foo1 -ivfsstatcache %t/stat.cache %s -Rmodule-build 2>&1
// RUN: cp %t/modules-cache/Foo.pcm %t/modules-to-compare/Foo-after.pcm

// RUN: diff %t/modules-to-compare/Foo-before.pcm %t/modules-to-compare/Foo-after.pcm

// ===
// Now introduce a different Foo module in the Foo2 directory which is before
// Foo1 in the search paths.
// RUN: echo 'void meow2(void);' > %t/Inputs/Foo2/foo.h
// RUN: echo 'module Foo { header "foo.h" }' > %t/Inputs/Foo2/module.map

// ===
// Because we're using the (now-outdated) stat cache, this compilation
// should still be using the first module. It will not see the new one
// which is earlier in the search paths.
// RUN: %clang_cc1 -cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -fsyntax-only -I %t/Inputs/Foo2 -I %t/Inputs/Foo1  -ivfsstatcache %t/stat.cache -Rmodule-build -Rmodule-import %s 2>&1
// RUN: cp %t/modules-cache/Foo.pcm %t/modules-to-compare/Foo-after.pcm

// RUN: diff %t/modules-to-compare/Foo-before.pcm %t/modules-to-compare/Foo-after.pcm

// ===
// Regenerate the stat cache for our Inputs directory
// RUN: clang-stat-cache -f %t/Inputs -o %t/stat.cache 2>&1

// ===
// Use the module and now see that we are recompiling the new one.
// RUN: %clang_cc1 -cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -fsyntax-only -I %t/Inputs/Foo2 -I %t/Inputs/Foo1 -ivfsstatcache %t/stat.cache -Rmodule-build %s 2>&1
// RUN: cp %t/modules-cache/Foo.pcm %t/modules-to-compare/Foo-after.pcm

// RUN: not diff %t/modules-to-compare/Foo-before.pcm %t/modules-to-compare/Foo-after.pcm
