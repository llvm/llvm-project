// RUN: rm -rf %t
// RUN: split-file %s %t
//
// At first build Stable.pcm that references Movable.framework from StableFrameworks.
// RUN: %clang_cc1 -fsyntax-only -F %t/JustBuilt -F %t/StableFrameworks %t/prepopulate-module-cache.m \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache
//
// Now add Movable.framework to JustBuilt.
// RUN: mkdir %t/JustBuilt
// RUN: cp -r %t/StableFrameworks/Movable.framework %t/JustBuilt/Movable.framework
//
// Load Movable.pcm at first for JustBuilt location and then in the same TU try to load transitively for StableFrameworks location.
// RUN: %clang_cc1 -fsyntax-only -F %t/JustBuilt -F %t/StableFrameworks %t/trigger-error.m \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache

// Test the case when a dependent module is found in a different location, so
// module cache has outdated information.

//--- StableFrameworks/Movable.framework/Headers/Movable.h
// empty

//--- StableFrameworks/Movable.framework/Modules/module.modulemap
framework module Movable {
  header "Movable.h"
  export *
}


//--- StableFrameworks/Stable.framework/Headers/Stable.h
#import <Movable/Movable.h>

//--- StableFrameworks/Stable.framework/Modules/module.modulemap
framework module Stable {
  header "Stable.h"
  export *
}


//--- prepopulate-module-cache.m
#import <Stable/Stable.h>

//--- trigger-error.m
#import <Movable/Movable.h>
#import <Stable/Stable.h>
