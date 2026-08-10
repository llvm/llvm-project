// RUN: rm -rf %t
// RUN: split-file %s %t

// This test checks that headers that are part of a module named by
// -fmodule-name= don't get included again if previously included from a PCH.

//--- include/module.modulemap
module Mod { header "Mod.h" }
//--- include/Mod.h
struct Symbol {};
//--- pch.h
#import "Mod.h"
//--- tu.c
#import "Mod.h" // expected-no-diagnostics

// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -fimplicit-module-maps -fmodule-name=Mod -I %t/include \
// RUN:   -emit-pch -x c-header %t/pch.h -o %t/pch.pch
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -fimplicit-module-maps -fmodule-name=Mod -I %t/include \
// RUN:   -fsyntax-only %t/tu.c -include-pch %t/pch.pch -verify
