// RUN: rm -rf %t.cache %tlocal.cache
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fmodules \
// RUN:   -fimplicit-module-maps -x c++ -emit-module \
// RUN:   -fmodules-cache-path=%t.cache \
// RUN:   -fmodule-name=pragma_pack %S/Inputs/module.map
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fmodules \
// RUN:   -fimplicit-module-maps -x c++ -verify \
// RUN:   -fmodules-cache-path=%t.cache \
// RUN:   -I%S/Inputs %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fmodules \
// RUN:   -fmodules-local-submodule-visibility \
// RUN:   -fimplicit-module-maps -x c++ -emit-module \
// RUN:   -fmodules-cache-path=%tlocal.cache \
// RUN:   -fmodule-name=pragma_pack %S/Inputs/module.map
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fmodules \
// RUN:   -fmodules-local-submodule-visibility \
// RUN:   -fimplicit-module-maps -x c++ -verify \
// RUN:   -fmodules-cache-path=%tlocal.cache \
// RUN:   -I%S/Inputs %s

// Check that we don't serialize pragma pack state until/unless including an
// empty file from the same module (but different submodule) has no effect.
#pragma pack (show) // expected-warning {{value of #pragma pack(show) == 8}}
#include "empty.h"
#pragma pack (show) // expected-warning {{value of #pragma pack(show) == 8}}
