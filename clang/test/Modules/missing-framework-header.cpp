// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

//--- frameworks/FW.framework/Modules/module.modulemap
framework module FW {
   umbrella header "FW.h"
   module * { export * }
}

//--- frameworks/FW.framework/Headers/FW.h
#include "One.h"
//--- frameworks/FW.framework/Headers/One.h
//--- frameworks/FW.framework/Headers/Two.h

//--- module.modulemap
module Mod { header "Mod.h" }
//--- Mod.h
#include "FW/Two.h"
//--- from_module.m
#include "Mod.h"

// RUN: %clang -fmodules -fmodules-cache-path=%t/cache \
// RUN: -iframework %t/frameworks -c %t/from_module.m -o %t/from_module.o \
// RUN:  2>&1 | FileCheck %s

// CHECK: warning: missing submodule 'FW.Two'

