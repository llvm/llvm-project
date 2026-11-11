// Tests -fmodules-ignore-search-path=P for implicitly-built modules: the path is
// dropped from the context hash of every module and physically removed from
// every module build, and kept only for the translation unit itself. It is the
// -fmodules-ignore-macro treatment applied to a search path rather than a macro,
// and like it, it applies to the regular context hash, so builds differing only
// in that path share modules.

// RUN: rm -rf %t && split-file %s %t

// Three "projects" that see the same modules but stage per-build generated
// headers at a different path each. Only the TU includes them; no module does.
// Under strict context hash those -I paths alone would fragment every module's
// hash.
//
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-strict-context-hash -fmodules-cache-path=%t/cache \
// RUN:   -I %t/common -I %t/ignored1 -fmodules-ignore-search-path=%t/ignored1 %t/tu.c
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-strict-context-hash -fmodules-cache-path=%t/cache \
// RUN:   -I %t/common -I %t/ignored2 -fmodules-ignore-search-path=%t/ignored2 %t/tu.c
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-strict-context-hash -fmodules-cache-path=%t/cache \
// RUN:   -I %t/common -I %t/ignored3 -fmodules-ignore-search-path=%t/ignored3 %t/tu.c

// All three projects share a single context hash for every module: exactly one
// pcm of each exists in the whole cache.
//
// RUN: ls %t/cache/*/Leaf-*.pcm | count 1
// RUN: ls %t/cache/*/Top-*.pcm | count 1

// Without the annotation the -I path feeds every module's context hash, so both
// modules fragment: one per project.
//
// RUN: rm -rf %t/cache
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-strict-context-hash -fmodules-cache-path=%t/cache \
// RUN:   -I %t/common -I %t/ignored1 %t/tu.c
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-strict-context-hash -fmodules-cache-path=%t/cache \
// RUN:   -I %t/common -I %t/ignored2 %t/tu.c
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-strict-context-hash -fmodules-cache-path=%t/cache \
// RUN:   -I %t/common -I %t/ignored3 %t/tu.c
// RUN: ls %t/cache/*/Leaf-*.pcm | count 3
// RUN: ls %t/cache/*/Top-*.pcm | count 3

// The path really is gone from the module builds, not merely unhashed: a module
// that does reach for a header only that path provides fails. That failure is the
// documented cost of asserting a path no module needs.
//
// RUN: rm -rf %t/cache
// RUN: not %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-strict-context-hash -fmodules-cache-path=%t/cache \
// RUN:   -I %t/common -I %t/ignored1 -fmodules-ignore-search-path=%t/ignored1 \
// RUN:   %t/tu-module-needs-ignored.c 2>&1 | FileCheck %s
//
// CHECK:      While building module 'NeedsIgnored' imported from {{.*}}tu-module-needs-ignored.c:1:
// CHECK-NEXT: In file included from <module-includes>:1:
// CHECK-NEXT: {{.*}}needs-ignored.h:1:10: fatal error: 'ignored.h' file not found
// CHECK:      {{.*}}tu-module-needs-ignored.c:1:10: fatal error: could not build module 'NeedsIgnored'

// And it is only the *ignored* path that goes: an unannotated -I path still
// works from inside a module, and still keys it.
//
// RUN: rm -rf %t/cache
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-strict-context-hash -fmodules-cache-path=%t/cache \
// RUN:   -I %t/common -I %t/ignored1 %t/tu-module-needs-ignored.c

//--- common/module.modulemap
module Leaf { header "leaf.h" }
module Top { header "top.h" }
module NeedsIgnored { header "needs-ignored.h" }

//--- common/leaf.h
//--- common/top.h
#include "leaf.h"
//--- common/needs-ignored.h
#include "ignored.h"

//--- ignored1/ignored.h
//--- ignored2/ignored.h
//--- ignored3/ignored.h

//--- tu.c
// Only the TU consumes the ignored path.
#include "ignored.h"
#include "top.h"

//--- tu-module-needs-ignored.c
#include "needs-ignored.h"
