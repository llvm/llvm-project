// RUN: rm -rf %t.cache
//
// RUN: %clang_cc1 -x objective-c-header -fmodules -F%S/Inputs/invalid-module-id \
// RUN:  -fmodule-implementation-of NC -fmodules-cache-path=%t.cache \
// RUN:  -fimplicit-module-maps \
// RUN:  -emit-pch %S/Inputs/invalid-module-id/NC-Prefix.pch -o %t.pch
//
// RUN: %clang_cc1 -x objective-c -fmodules -F%S/Inputs/invalid-module-id \
// RUN:  -fmodule-implementation-of NC -fmodules-cache-path=%t.cache \
// RUN:  -fimplicit-module-maps -include-pch %t.pch %s -fsyntax-only

#import <NC/NULog.h>
#import <NC/NUGeometry.h>
