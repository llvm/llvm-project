// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x c++ -Rmodule-build -I%S/Inputs/module-map-path-hash -fmodules-cache-path=%t -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=BUILD
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x c++ -Rmodule-build -I%S/Inputs//module-map-path-hash -fmodules-cache-path=%t -fsyntax-only %s 2>&1 | FileCheck -allow-empty %s --check-prefix=NOBUILD
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x c++ -Rmodule-build -I%S/Inputs/./module-map-path-hash -fmodules-cache-path=%t -fsyntax-only %s 2>&1 | FileCheck -allow-empty %s --check-prefix=NOBUILD
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x c++ -Rmodule-build -I%S/Inputs/../Inputs/module-map-path-hash -fmodules-cache-path=%t -fsyntax-only %s 2>&1 | FileCheck -allow-empty %s --check-prefix=NOBUILD

#include "a.h"

// BUILD: remark: building module
// NOBUILD-NOT: remark: building module
