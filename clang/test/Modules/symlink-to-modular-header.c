// REQUIRES: symlinks

// Test that we warn when including a symlink to a modular header that isn't
// covered by a module map at the symlink's location.

// RUN: rm -rf %t
// RUN: split-file %s %t

// Create the symlink directories and symlinks.
// RUN: mkdir -p %t/covered %t/uncovered
// RUN: ln -s %t/ModuleA/A.h %t/covered/A.h
// RUN: ln -s %t/ModuleA/A.h %t/uncovered/A.h

// Including the symlink that IS covered should not warn.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t/covered -I%t/ModuleA \
// RUN:   -fmodules-cache-path=%t/cache -fsyntax-only %t/tu-covered.m \
// RUN:   -Wmmap-deprecated-symlink-to-modular-header -verify

// Including the symlink that is NOT covered should warn.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t/uncovered -I%t/ModuleA \
// RUN:   -fmodules-cache-path=%t/cache -fsyntax-only %t/tu-uncovered.m \
// RUN:   -Wmmap-deprecated-symlink-to-modular-header -verify

// Same test with lazy module map loading.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t/uncovered -I%t/ModuleA \
// RUN:   -fmodules-cache-path=%t/cache2 -fmodules-lazy-load-module-maps \
// RUN:   -fsyntax-only %t/tu-uncovered.m \
// RUN:   -Wmmap-deprecated-symlink-to-modular-header -verify

// Test the diagnostic when the module is loaded from a PCM. Import B which
// transitively builds A into a PCM, then include A.h via the uncovered
// symlink so A is loaded from the PCM.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t \
// RUN:   -fmodules-cache-path=%t/cache3 -fsyntax-only %t/tu-uncovered-pcm.m \
// RUN:   -Wmmap-deprecated-symlink-to-modular-header \
// RUN:   -fno-modules-check-relocated -Rmodule-map -verify

//--- ModuleA/module.modulemap
module A {
  header "A.h"
}

//--- ModuleA/A.h

//--- ModuleB/module.modulemap
module B {
  header "B.h"
  export *
}

//--- ModuleB/B.h
#include "ModuleA/A.h"

//--- covered/module.modulemap
module SymlinkA {
  header "A.h"
}

//--- tu-covered.m
// expected-no-diagnostics
#include "A.h"

//--- tu-uncovered.m
#pragma clang __debug module_lookup A
#include "A.h"

// expected-warning-re@* {{may be a symlink to '{{.*}}' owned by module 'A'}}
// expected-note@ModuleA/module.modulemap:1 {{module defined here}}

//--- tu-uncovered-pcm.m
#include "ModuleB/B.h"
#include "uncovered/A.h"

// expected-warning-re@* {{may be a symlink to '{{.*}}' owned by module 'A'}}
// expected-note@ModuleA/module.modulemap:1 {{module defined here}}
// expected-remark-re@* {{loading modulemap '{{.*}}ModuleB/module.modulemap'}}
// expected-remark-re@* {{loading modulemap '{{.*}}module.modulemap'}}
// expected-remark-re@* {{loading modulemap '{{.*}}pthread.modulemap'}}

// RUN: ln -s pthread/pthread.h %t/pthread.h
// RUN: %clang_cc1 -fmodules-cache-path=%t/modules -fmodules -fimplicit-module-maps -I %t %t/tu.c -fsyntax-only \
// RUN:   -Wmmap-deprecated-symlink-to-modular-header -verify

//--- module.modulemap
extern module pthread "pthread.modulemap"

//--- pthread.modulemap
module pthread { header "pthread/pthread.h" }

//--- pthread/pthread.h
typedef int p;

//--- tu.c
#include <pthread.h>
p P = 4;

// expected-warning-re@* {{may be a symlink to '{{.*}}' owned by module 'pthread'}}
// expected-note@pthread.modulemap:1 {{module defined here}}
