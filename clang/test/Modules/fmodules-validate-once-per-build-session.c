#include "foo.h"
#include "bar.h"

// Clear the module cache.
// RUN: rm -rf %t
// RUN: mkdir -p %t/Inputs
// RUN: mkdir -p %t/modules-to-compare

// ===
// Create a module.  We will use -I or -isystem to determine whether to treat
// foo.h as a system header.
// RUN: echo 'void meow(void);' > %t/Inputs/foo.h
// RUN: echo 'void woof(void);' > %t/Inputs/bar.h
// RUN: echo 'module Foo { header "foo.h" }' > %t/Inputs/module.modulemap
// RUN: echo 'extern module Bar "bar.modulemap"' >> %t/Inputs/module.modulemap
// RUN: echo 'module Bar { header "bar.h" }' > %t/Inputs/bar.modulemap

// ===
// Compile the module.
// RUN: %clang_cc1 -cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -fsyntax-only -isystem %t/Inputs -fmodules-validate-system-headers -fbuild-session-timestamp=1390000000 -fmodules-validate-once-per-build-session %s
// RUN: %clang_cc1 -cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache-user -fsyntax-only -I %t/Inputs -fmodules-validate-system-headers -fbuild-session-timestamp=1390000000 -fmodules-validate-once-per-build-session %s
// RUN: %clang_cc1 -cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache-user-no-force -fsyntax-only -I %t/Inputs -fno-modules-force-validate-user-headers -fmodules-validate-system-headers -fbuild-session-timestamp=1390000000 -fmodules-validate-once-per-build-session %s
// RUN: ls -R %t/modules-cache | grep Foo.pcm.timestamp
// RUN: ls -R %t/modules-cache | grep Bar.pcm.timestamp
// RUN: ls -R %t/modules-cache-user | grep Foo.pcm.timestamp
// RUN: ls -R %t/modules-cache-user | grep Bar.pcm.timestamp
// RUN: ls -R %t/modules-cache-user-no-force | grep Foo.pcm.timestamp
// RUN: ls -R %t/modules-cache-user-no-force | grep Bar.pcm.timestamp
// RUN: cp %t/modules-cache/Foo.pcm %t/modules-to-compare/Foo-before.pcm
// RUN: cp %t/modules-cache/Bar.pcm %t/modules-to-compare/Bar-before.pcm
// RUN: cp %t/modules-cache-user/Foo.pcm %t/modules-to-compare/Foo-before-user.pcm
// RUN: cp %t/modules-cache-user/Bar.pcm %t/modules-to-compare/Bar-before-user.pcm
// RUN: cp %t/modules-cache-user-no-force/Foo.pcm %t/modules-to-compare/Foo-before-user-no-force.pcm
// RUN: cp %t/modules-cache-user-no-force/Bar.pcm %t/modules-to-compare/Bar-before-user-no-force.pcm

// ===
// Use it, and make sure that we did not recompile it.
// RUN: %clang_cc1 -cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -fsyntax-only -isystem %t/Inputs -fmodules-validate-system-headers -fbuild-session-timestamp=1390000000 -fmodules-validate-once-per-build-session %s
// RUN: %clang_cc1 -cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache-user -fsyntax-only -I %t/Inputs -fmodules-validate-system-headers -fbuild-session-timestamp=1390000000 -fmodules-validate-once-per-build-session %s
// RUN: %clang_cc1 -cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache-use-no-force -fsyntax-only -I %t/Inputs -fno-modules-force-validate-user-headers -fmodules-validate-system-headers -fbuild-session-timestamp=1390000000 -fmodules-validate-once-per-build-session %s
// RUN: ls -R %t/modules-cache | grep Foo.pcm.timestamp
// RUN: ls -R %t/modules-cache | grep Bar.pcm.timestamp
// RUN: ls -R %t/modules-cache-user | grep Foo.pcm.timestamp
// RUN: ls -R %t/modules-cache-user | grep Bar.pcm.timestamp
// RUN: ls -R %t/modules-cache-user-no-force | grep Foo.pcm.timestamp
// RUN: ls -R %t/modules-cache-user-no-force | grep Bar.pcm.timestamp
// RUN: cp %t/modules-cache/Foo.pcm %t/modules-to-compare/Foo-after.pcm
// RUN: cp %t/modules-cache/Bar.pcm %t/modules-to-compare/Bar-after.pcm
// RUN: cp %t/modules-cache-user/Foo.pcm %t/modules-to-compare/Foo-after-user.pcm
// RUN: cp %t/modules-cache-user/Bar.pcm %t/modules-to-compare/Bar-after-user.pcm
// RUN: cp %t/modules-cache-user-no-force/Foo.pcm %t/modules-to-compare/Foo-after-user-no-force.pcm
// RUN: cp %t/modules-cache-user-no-force/Bar.pcm %t/modules-to-compare/Bar-after-user-no-force.pcm

// RUN: diff %t/modules-to-compare/Foo-before.pcm %t/modules-to-compare/Foo-after.pcm
// RUN: diff %t/modules-to-compare/Bar-before.pcm %t/modules-to-compare/Bar-after.pcm
// RUN: diff %t/modules-to-compare/Foo-before-user.pcm %t/modules-to-compare/Foo-after-user.pcm
// RUN: diff %t/modules-to-compare/Bar-before-user.pcm %t/modules-to-compare/Bar-after-user.pcm
// RUN: diff %t/modules-to-compare/Foo-before-user-no-force.pcm %t/modules-to-compare/Foo-after-user-no-force.pcm
// RUN: diff %t/modules-to-compare/Bar-before-user-no-force.pcm %t/modules-to-compare/Bar-after-user-no-force.pcm

// ===
// Change the sources.
// RUN: echo 'void meow2(void);' > %t/Inputs/foo.h
// RUN: echo 'module Bar { header "bar.h" export * }' > %t/Inputs/bar.modulemap

// ===
// Use the module, and make sure that we did not recompile it if foo.h or
// module.modulemap are system files or user files with force validation disabled,
// even though the sources changed.
// RUN: %clang_cc1 -cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -fsyntax-only -isystem %t/Inputs -fmodules-validate-system-headers -fbuild-session-timestamp=1390000000 -fmodules-validate-once-per-build-session %s
// RUN: %clang_cc1 -cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache-user -fsyntax-only -I %t/Inputs -fmodules-validate-system-headers -fbuild-session-timestamp=1390000000 -fmodules-validate-once-per-build-session %s
// RUN: %clang_cc1 -cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache-user-no-force -fsyntax-only -I %t/Inputs -fno-modules-force-validate-user-headers -fmodules-validate-system-headers -fbuild-session-timestamp=1390000000 -fmodules-validate-once-per-build-session %s
// RUN: ls -R %t/modules-cache | grep Foo.pcm.timestamp
// RUN: ls -R %t/modules-cache | grep Bar.pcm.timestamp
// RUN: ls -R %t/modules-cache-user | grep Foo.pcm.timestamp
// RUN: ls -R %t/modules-cache-user | grep Bar.pcm.timestamp
// RUN: ls -R %t/modules-cache-user-no-force | grep Foo.pcm.timestamp
// RUN: ls -R %t/modules-cache-user-no-force | grep Bar.pcm.timestamp
// RUN: cp %t/modules-cache/Foo.pcm %t/modules-to-compare/Foo-after.pcm
// RUN: cp %t/modules-cache/Bar.pcm %t/modules-to-compare/Bar-after.pcm
// RUN: cp %t/modules-cache-user/Foo.pcm %t/modules-to-compare/Foo-after-user.pcm
// RUN: cp %t/modules-cache-user/Bar.pcm %t/modules-to-compare/Bar-after-user.pcm
// RUN: cp %t/modules-cache-user-no-force/Foo.pcm %t/modules-to-compare/Foo-after-user-no-force.pcm
// RUN: cp %t/modules-cache-user-no-force/Bar.pcm %t/modules-to-compare/Bar-after-user-no-force.pcm

// RUN: diff %t/modules-to-compare/Foo-before.pcm %t/modules-to-compare/Foo-after.pcm
// RUN: diff %t/modules-to-compare/Bar-before.pcm %t/modules-to-compare/Bar-after.pcm
// When foo.h is an user header, we will validate it by default.
// RUN: not diff %t/modules-to-compare/Foo-before-user.pcm %t/modules-to-compare/Foo-after-user.pcm
// RUN: not diff %t/modules-to-compare/Bar-before-user.pcm %t/modules-to-compare/Bar-after-user.pcm
// When foo.h is an user header, we will not validate it if force validation is turned off.
// RUN: diff %t/modules-to-compare/Foo-before-user-no-force.pcm %t/modules-to-compare/Foo-after-user-no-force.pcm
// RUN: diff %t/modules-to-compare/Bar-before-user-no-force.pcm %t/modules-to-compare/Bar-after-user-no-force.pcm

// ===
// Recompile the module if the today's date is before 01 January 2100.
// RUN: %clang_cc1 -cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -fsyntax-only -isystem %t/Inputs -fmodules-validate-system-headers -fbuild-session-timestamp=4102441200 -fmodules-validate-once-per-build-session %s
// RUN: %clang_cc1 -cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache-user -fsyntax-only -I %t/Inputs -fno-modules-force-validate-user-headers -fmodules-validate-system-headers -fbuild-session-timestamp=4102441200 -fmodules-validate-once-per-build-session %s
// RUN: %clang_cc1 -cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache-user-no-force -fsyntax-only -I %t/Inputs -fno-modules-force-validate-user-headers -fmodules-validate-system-headers -fbuild-session-timestamp=4102441200 -fmodules-validate-once-per-build-session %s
// RUN: ls -R %t/modules-cache | grep Foo.pcm.timestamp
// RUN: ls -R %t/modules-cache | grep Bar.pcm.timestamp
// RUN: ls -R %t/modules-cache-user | grep Foo.pcm.timestamp
// RUN: ls -R %t/modules-cache-user | grep Bar.pcm.timestamp
// RUN: ls -R %t/modules-cache-user-no-force | grep Foo.pcm.timestamp
// RUN: ls -R %t/modules-cache-user-no-force | grep Bar.pcm.timestamp
// RUN: cp %t/modules-cache/Foo.pcm %t/modules-to-compare/Foo-after.pcm
// RUN: cp %t/modules-cache/Bar.pcm %t/modules-to-compare/Bar-after.pcm
// RUN: cp %t/modules-cache-user/Foo.pcm %t/modules-to-compare/Foo-after-user.pcm
// RUN: cp %t/modules-cache-user/Bar.pcm %t/modules-to-compare/Bar-after-user.pcm
// RUN: cp %t/modules-cache-user-no-force/Foo.pcm %t/modules-to-compare/Foo-after-user-no-force.pcm
// RUN: cp %t/modules-cache-user-no-force/Bar.pcm %t/modules-to-compare/Bar-after-user-no-force.pcm

// RUN: not diff %t/modules-to-compare/Foo-before.pcm %t/modules-to-compare/Foo-after.pcm
// RUN: not diff %t/modules-to-compare/Bar-before.pcm %t/modules-to-compare/Bar-after.pcm
// RUN: not diff %t/modules-to-compare/Foo-before-user.pcm %t/modules-to-compare/Foo-after-user.pcm
// RUN: not diff %t/modules-to-compare/Bar-before-user.pcm %t/modules-to-compare/Bar-after-user.pcm
// RUN: not diff %t/modules-to-compare/Foo-before-user-no-force.pcm %t/modules-to-compare/Foo-after-user-no-force.pcm
// RUN: not diff %t/modules-to-compare/Bar-before-user-no-force.pcm %t/modules-to-compare/Bar-after-user-no-force.pcm
