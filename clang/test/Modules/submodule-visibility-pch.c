// Verify that the use of a PCH does not accidentally make modules from the PCH
// visible to submodules when using -fmodules-local-submodule-visibility
// and -fmodule-name to trigger a textual include.

// RUN: rm -rf %t
// RUN: split-file %s %t

// First check that it works with a header

// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache \
// RUN:   -fmodules-local-submodule-visibility -fimplicit-module-maps \
// RUN:   -fmodule-name=Mod \
// RUN:   %t/tu.c -include %t/prefix.h -I %t -verify

// Now with a PCH

// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache \
// RUN:   -fmodules-local-submodule-visibility -fimplicit-module-maps \
// RUN:   -x c-header %t/prefix.h -emit-pch -o %t/prefix.pch -I %t

// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache \
// RUN:   -fmodules-local-submodule-visibility -fimplicit-module-maps \
// RUN:   -fmodule-name=Mod \
// RUN:   %t/tu.c -include-pch %t/prefix.pch -I %t -verify

//--- module.modulemap
module ModViaPCH { header "ModViaPCH.h" }
module ModViaInclude { header "ModViaInclude.h" }
module Mod { header "Mod.h" }
module SomeOtherMod { header "SomeOtherMod.h" }

//--- ModViaPCH.h
#define ModViaPCH 1

//--- ModViaInclude.h
#define ModViaInclude 2

//--- SomeOtherMod.h
// empty

//--- Mod.h
#include "SomeOtherMod.h"
#ifdef ModViaPCH
#error "Visibility violation ModViaPCH"
#endif
#ifdef ModViaInclude
#error "Visibility violation ModViaInclude"
#endif

//--- prefix.h
#include "ModViaPCH.h"

//--- tu.c
#include "ModViaInclude.h"
#include "Mod.h"
// expected-no-diagnostics
