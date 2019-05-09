#include "foo.h"

// Clear the module cache.
// RUN: rm -rf %t
// RUN: mkdir -p %t/Inputs
// RUN: mkdir -p %t/modules-to-compare

// ===
// Create a module.  We will use -I or -isystem to determine whether to treat
// foo.h as a system header.
// RUN: echo 'void meow(void);' > %t/Inputs/foo.h
// RUN: echo 'module Foo { header "foo.h" }' > %t/Inputs/module.map

// ===
// Compile the module.
// RUN: %clang_cc1 -cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -fsyntax-only -isystem %t/Inputs -fbuild-session-timestamp=1390000000 -fmodules-validate-once-per-build-session %s
// RUN: %clang_cc1 -cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache-user -fsyntax-only -I %t/Inputs -fbuild-session-timestamp=1390000000 -fmodules-validate-once-per-build-session %s
// RUN: ls -R %t/modules-cache | grep Foo.pcm.timestamp
// RUN: ls -R %t/modules-cache-user | grep Foo.pcm.timestamp
// RUN: cp %t/modules-cache/Foo.pcm %t/modules-to-compare/Foo-before.pcm
// RUN: cp %t/modules-cache-user/Foo.pcm %t/modules-to-compare/Foo-before-user.pcm

// ===
// Use it, and make sure that we did not recompile it.
// RUN: %clang_cc1 -cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -fsyntax-only -isystem %t/Inputs -fbuild-session-timestamp=1390000000 -fmodules-validate-once-per-build-session %s
// RUN: %clang_cc1 -cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache-user -fsyntax-only -I %t/Inputs -fbuild-session-timestamp=1390000000 -fmodules-validate-once-per-build-session %s
// RUN: ls -R %t/modules-cache | grep Foo.pcm.timestamp
// RUN: ls -R %t/modules-cache-user | grep Foo.pcm.timestamp
// RUN: cp %t/modules-cache/Foo.pcm %t/modules-to-compare/Foo-after.pcm
// RUN: cp %t/modules-cache-user/Foo.pcm %t/modules-to-compare/Foo-after-user.pcm

// RUN: diff %t/modules-to-compare/Foo-before.pcm %t/modules-to-compare/Foo-after.pcm
// RUN: diff %t/modules-to-compare/Foo-before-user.pcm %t/modules-to-compare/Foo-after-user.pcm

// ===
// Change the sources.
// RUN: echo 'void meow2(void);' > %t/Inputs/foo.h

// ===
// Use the module, and make sure that we did not recompile it if foo.h is a
// system header, even though the sources changed.
// RUN: %clang_cc1 -cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -fsyntax-only -isystem %t/Inputs -fbuild-session-timestamp=1390000000 -fmodules-validate-once-per-build-session %s
// RUN: %clang_cc1 -cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache-user -fsyntax-only -I %t/Inputs -fbuild-session-timestamp=1390000000 -fmodules-validate-once-per-build-session %s
// RUN: ls -R %t/modules-cache | grep Foo.pcm.timestamp
// RUN: ls -R %t/modules-cache-user | grep Foo.pcm.timestamp
// RUN: cp %t/modules-cache/Foo.pcm %t/modules-to-compare/Foo-after.pcm
// RUN: cp %t/modules-cache-user/Foo.pcm %t/modules-to-compare/Foo-after-user.pcm

// RUN: diff %t/modules-to-compare/Foo-before.pcm %t/modules-to-compare/Foo-after.pcm
// When foo.h is a user header, we will always validate it.
// RUN: not diff %t/modules-to-compare/Foo-before-user.pcm %t/modules-to-compare/Foo-after-user.pcm

// ===
// Recompile the module if the today's date is before 01 January 2100.
// RUN: %clang_cc1 -cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -fsyntax-only -isystem %t/Inputs -fbuild-session-timestamp=4102441200 -fmodules-validate-once-per-build-session %s
// RUN: ls -R %t/modules-cache | grep Foo.pcm.timestamp
// RUN: cp %t/modules-cache/Foo.pcm %t/modules-to-compare/Foo-after.pcm

// RUN: not diff %t/modules-to-compare/Foo-before.pcm %t/modules-to-compare/Foo-after.pcm
