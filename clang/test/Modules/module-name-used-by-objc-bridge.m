// RUN: rm -rf %t && mkdir %t

// RUN: %clang_cc1 -fmodules -x objective-c -emit-module -fmodule-name=InterfaceBridge \
// RUN:   %S/Inputs/module-name-used-by-objc-bridge/module.modulemap -o %t/InterfaceBridge.pcm

// RUN: %clang_cc1 -fmodules -x objective-c -emit-module -fmodule-name=Interface \
// RUN:   %S/Inputs/module-name-used-by-objc-bridge/module.modulemap -o %t/Interface.pcm

// Check that the `-fmodule-file=<name>=<path>` form succeeds:
// RUN: %clang_cc1 -fmodules -fsyntax-only %s -I %S/Inputs/module-name-used-by-objc-bridge \
// RUN:   -fmodule-file=InterfaceBridge=%t/InterfaceBridge.pcm -fmodule-file=Interface=%t/Interface.pcm \
// RUN:   -fmodule-map-file=%S/Inputs/module-name-used-by-objc-bridge/module.modulemap -verify

// Check that the `-fmodule-file=<path>` form succeeds:
// RUN: %clang_cc1 -fmodules -fsyntax-only %s -I %S/Inputs/module-name-used-by-objc-bridge \
// RUN:   -fmodule-file=%t/InterfaceBridge.pcm -fmodule-file=%t/Interface.pcm \
// RUN:   -fmodule-map-file=%S/Inputs/module-name-used-by-objc-bridge/module.modulemap -verify

#import "InterfaceBridge.h"
#import "Interface.h"

@interface Interface (User)
@end

// expected-no-diagnostics
