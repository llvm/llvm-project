// RUN: rm -rf %t

// Produce an error if a module is needed, but not found.
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:     -fmodule-map-file=%S/Inputs/no-implicit-builds/b.modulemap \
// RUN:     -fno-implicit-modules %s -verify
//
// Same thing if we're running -cc1 and no module cache path has been provided.
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps \
// RUN:     -fmodule-map-file=%S/Inputs/no-implicit-builds/b.modulemap \
// RUN:     %s -verify

// Compile the module and put it into the cache.
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:     -fmodule-map-file=%S/Inputs/no-implicit-builds/b.modulemap \
// RUN:     %s -Rmodule-build 2>&1 | FileCheck --check-prefix=CHECK-CACHE-BUILD %s
// CHECK-CACHE-BUILD: {{building module 'b'}}

// Produce an error if a module is found in the cache but implicit modules is off.
// Note that the command line must match the command line for the first check, otherwise
// this check might not find the module in the cache and trivially succeed.
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:     -fmodule-map-file=%S/Inputs/no-implicit-builds/b.modulemap \
// RUN:     %s -Rmodule-build -fno-implicit-modules -verify

// Verify that we can still pass the module via -fmodule-file when implicit modules
// are switched off:
// - First, explicitly compile the module:
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodule-name=b -o %t/b.pcm \
// RUN:     -emit-module %S/Inputs/no-implicit-builds/b.modulemap \
// RUN:     -fno-implicit-modules
//
// - Next, verify that we can load it:
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodule-file=%t/b.pcm \
// RUN:     -fmodule-map-file=%S/Inputs/no-implicit-builds/b.modulemap \
// RUN:     -fno-implicit-modules %s

#include "Inputs/no-implicit-builds/b.h"  // expected-error {{is needed but has not been provided}}
