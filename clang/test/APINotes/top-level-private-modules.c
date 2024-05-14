// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fapinotes-modules -Wno-private-module -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -verify

#include <PrivateLib.h>
#include <TopLevelPrivateKit/TopLevelPrivateKit_Private.h>

void *testPlain = PrivateLib; // expected-error {{initializing 'void *' with an expression of incompatible type 'float'}}
void *testFramework = TopLevelPrivateKit_Private; // expected-error {{initializing 'void *' with an expression of incompatible type 'float'}}
