// RUN: rm -rf %t && mkdir -p %t
// RUN: split-file %s %t

//--- module.modulemap
module Mod { header "mod.h" }
//--- mod.h
//--- tu.c
#include "mod.h"

// Without any extra compiler flags, mismatched diagnostic options trigger recompilation of modules.
//
// RUN: %clang_cc1 -fmodules -fmodule-map-file=%t/module.modulemap -fmodules-cache-path=%t/cache1 -fdisable-module-hash \
// RUN:   -fsyntax-only %t/tu.c -Wnon-modular-include-in-module
// RUN: %clang_cc1 -fmodules -fmodule-map-file=%t/module.modulemap -fmodules-cache-path=%t/cache1 -fdisable-module-hash \
// RUN:   -fsyntax-only %t/tu.c -Werror=non-modular-include-in-module -Rmodule-build 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DID-REBUILD
// DID-REBUILD: remark: building module 'Mod'

// When skipping serialization of diagnostic options, mismatches cannot be detected, old PCM file gets reused.
//
// RUN: %clang_cc1 -fmodules -fmodule-map-file=%t/module.modulemap -fmodules-cache-path=%t/cache2 -fdisable-module-hash \
// RUN:   -fsyntax-only %t/tu.c -fmodules-skip-diagnostic-options -Wnon-modular-include-in-module
// RUN: %clang_cc1 -fmodules -fmodule-map-file=%t/module.modulemap -fmodules-cache-path=%t/cache2 -fdisable-module-hash \
// RUN:   -fsyntax-only %t/tu.c -fmodules-skip-diagnostic-options -Werror=non-modular-include-in-module -Rmodule-build 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DID-REUSE --allow-empty
// DID-REUSE-NOT: remark: building module 'Mod'
//
// RUN: %clang_cc1 -module-file-info %t/cache2/Mod.pcm | FileCheck %s --check-prefix=NO-DIAG-OPTS
// NO-DIAG-OPTS-NOT: Diagnostic flags:

// When disabling validation of diagnostic options, mismatches are not checked for, old PCM file gets reused.
//
// RUN: %clang_cc1 -fmodules -fmodule-map-file=%t/module.modulemap -fmodules-cache-path=%t/cache3 -fdisable-module-hash \
// RUN:   -fsyntax-only %t/tu.c -fmodules-disable-diagnostic-validation -Wnon-modular-include-in-module
// RUN: %clang_cc1 -fmodules -fmodule-map-file=%t/module.modulemap -fmodules-cache-path=%t/cache3 -fdisable-module-hash \
// RUN:   -fsyntax-only %t/tu.c -fmodules-disable-diagnostic-validation -Werror=non-modular-include-in-module -Rmodule-build 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DID-REUSE --allow-empty
//
// RUN: %clang_cc1 -module-file-info %t/cache3/Mod.pcm | FileCheck %s --check-prefix=OLD-DIAG-OPTS
// OLD-DIAG-OPTS: Diagnostic flags:
// OLD-DIAG-OPTS-NEXT: -Wnon-modular-include-in-module
