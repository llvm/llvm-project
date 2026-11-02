// Check an explicit module's command line names the module map of every
// dependency it is given with -fmodule-file.

// In this example, EntryPoint imports DirectOnly first, and loading DirectOnly's
//  PCM populates DirectAndTransitive into memory.
// Then when EntryPoint imports DirectAndTransitive (through @import) itself, the
//  module is already known, so the lookup short-circuits and never searches
//  DirectAndTransitive's module map in EntryPoint's CompilerInstance as an input file.
// The scan should still register DirectAndTransitive's modulemap as a
//  direct dependency to ensure the resulting explicit module invocation is stable
//  as additional invalidation or relocation checks can cause EntryPoint's
//  CompilerInstance to observe that modulemap as an input file dependency.

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: clang-scan-deps -format experimental-full -- \
// RUN:   %clang -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:     -I %t/entrypoint -I %t/directonly -I %t/directandtransitive -c %t/tu.m -o %t/tu.o \
// RUN:   > %t/result.json

// RUN: %deps-to-rsp %t/result.json --module-name=EntryPoint > %t/EntryPoint.rsp
// RUN: cat %t/EntryPoint.rsp | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK: -fmodule-map-file=[[PREFIX]]/directonly/module.modulemap
// CHECK: -fmodule-map-file=[[PREFIX]]/directandtransitive/module.modulemap
// CHECK: -fmodule-file=DirectAndTransitive={{[^"]*}}.pcm
// CHECK: -fmodule-file=DirectOnly={{[^"]*}}.pcm

// Verify round-trip compilation passes.
// RUN: %deps-to-rsp %t/result.json --module-name=DirectAndTransitive > %t/DirectAndTransitive.rsp
// RUN: %deps-to-rsp %t/result.json --module-name=DirectOnly > %t/DirectOnly.rsp
// RUN: %deps-to-rsp %t/result.json --tu-index=0 > %t/tu.rsp
// RUN: %clang @%t/DirectAndTransitive.rsp
// RUN: %clang @%t/DirectOnly.rsp
// RUN: %clang @%t/EntryPoint.rsp
// RUN: %clang @%t/tu.rsp

//--- directandtransitive/module.modulemap
module DirectAndTransitive { header "direct-and-transitive.h" export * }

//--- directandtransitive/direct-and-transitive.h
int directAndTransitive(void);

//--- directonly/module.modulemap
module DirectOnly { header "direct-only.h" export * }

//--- directonly/direct-only.h
#include "direct-and-transitive.h"
int directOnly(void);

//--- entrypoint/module.modulemap
module EntryPoint { header "entry-point.h" export * }

//--- entrypoint/entry-point.h
#include "direct-only.h"
@import DirectAndTransitive;
int entryPoint(void);

//--- tu.m
#include "entry-point.h"
int tu(void) { return entryPoint(); }
