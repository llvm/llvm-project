// This tests that after a fatal module loader error, we do not continue parsing.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: mkdir %t/ModulesCache
// RUN: touch %t/ModulesCache/Unusable.pcm

// RUN: not clang-scan-deps -format experimental-full -mode preprocess-dependency-directives -- \
// RUN:   %clang -fmodules -fimplicit-modules -Xclang -fdisable-module-hash -fmodules-cache-path=%t/ModulesCache \
// RUN:   -F %t/Frameworks -c %t/test.m -o %t/test.o
// RUN: ls %t/ModulesCache | not grep AfterUnusable

//--- Frameworks/Unusable.framework/Headers/Unusable.h
// empty
//--- Frameworks/Unusable.framework/Modules/module.modulemap
framework module Unusable { header "Unusable.h" }

//--- Frameworks/AfterUnusable.framework/Headers/AfterUnusable.h
// empty
//--- Frameworks/AfterUnusable.framework/Modules/module.modulemap
framework module AfterUnusable { header "AfterUnusable.h" }

//--- Frameworks/Importer.framework/Headers/Importer.h
#import <Importer/ImportUnusable.h>
// Parsing should have stopped and we should not handle AfterUnusable.
#import <AfterUnusable/AfterUnusable.h>

//--- Frameworks/Importer.framework/Headers/ImportUnusable.h
// It is important that this header is a submodule.
#import <Unusable/Unusable.h>
// Parsing should have stopped and we should not handle AfterUnusable.
#import <AfterUnusable/AfterUnusable.h>

//--- Frameworks/Importer.framework/Modules/module.modulemap
framework module Importer {
  umbrella header "Importer.h"
  module * { export * }
  export *
}

//--- test.m
#import <Importer/Importer.h>
