// This test checks that with -fmodules-single-module-parse-mode, no modules get
// compiled into PCM files from any of the import syntax Clang supports.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: mkdir %t/cache

// With -fmodules-single-module-parse-mode, no modules get compiled.
// RUN: %clang_cc1 -x objective-c -fmodules -fmodules-cache-path=%t/cache \
// RUN:   -emit-module %t/module.modulemap -fmodule-name=Mod -o %t/Mod.pcm \
// RUN:   -fmodules-single-module-parse-mode
// RUN: find %t/cache -name "*.pcm" | count 0

// Without -fmodules-single-module-parse-mode, loaded modules get compiled.
// RUN: %clang_cc1 -x objective-c -fmodules -fmodules-cache-path=%t/cache \
// RUN:   -emit-module %t/module.modulemap -fmodule-name=Mod -o %t/Mod.pcm
// RUN: find %t/cache -name "*.pcm" | count 5

//--- module.modulemap
module Mod { header "Mod.h" }
module Load1 { header "Load1.h" }
module Load2 { header "Load2.h" }
module Load3 { header "Load3.h" }
module Load4 { header "Load4.h" }
module Load5 { header "Load5.h" }
//--- Mod.h
#include "Load1.h"
#import "Load2.h"
@import Load3;
#pragma clang module import Load4
#pragma clang module load Load5
//--- Load1.h
//--- Load2.h
//--- Load3.h
//--- Load4.h
//--- Load5.h
