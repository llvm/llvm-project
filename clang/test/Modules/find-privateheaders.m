// RUN: rm -rf %t.cache
// RUN: %clang_cc1 -fmodules -fsyntax-only -F%S/Inputs -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t.cache -Wno-private-module -DBUILD_PUBLIC -verify %s
// RUN: rm -rf %t.cache
// RUN: %clang_cc1 -fmodules -fsyntax-only -F%S/Inputs -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t.cache -Wno-private-module -verify %s
//expected-no-diagnostics

#ifdef BUILD_PUBLIC
#import "Main/Main.h"
#else
#import "MainA/MainPriv.h"
#endif
