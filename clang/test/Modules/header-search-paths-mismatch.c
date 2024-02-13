// RUN: rm -rf %t && mkdir -p %t
// RUN: split-file %s %t

//--- module.modulemap
module Mod { header "mod.h" }
//--- mod.h
//--- tu.c
#include "mod.h"

//--- one/foo.h
//--- two/foo.h

// By default, mismatched header search paths are ignored, old PCM file gets reused.
//
// RUN: %clang_cc1 -fmodules -fmodule-map-file=%t/module.modulemap -fmodules-cache-path=%t/cache1 -fdisable-module-hash \
// RUN:   -fsyntax-only %t/tu.c -I %t/one
// RUN: %clang_cc1 -fmodules -fmodule-map-file=%t/module.modulemap -fmodules-cache-path=%t/cache1 -fdisable-module-hash \
// RUN:   -fsyntax-only %t/tu.c -I %t/two -Rmodule-build 2>&1 \
// RUN:   | FileCheck %s --allow-empty --check-prefix=DID-REUSE
// DID-REUSE-NOT: remark: building module 'Mod'
//
// RUN: %clang_cc1 -module-file-info %t/cache1/Mod.pcm | FileCheck %s --check-prefix=HS-PATHS
// HS-PATHS:      Header search paths:
// HS-PATHS-NEXT:   User entries:
// HS-PATHS-NEXT:     one

// When skipping serialization of header search paths, mismatches cannot be detected, old PCM file gets reused.
//
// RUN: %clang_cc1 -fmodules -fmodule-map-file=%t/module.modulemap -fmodules-cache-path=%t/cache2 -fdisable-module-hash \
// RUN:   -fsyntax-only %t/tu.c -fmodules-skip-header-search-paths -I %t/one
// RUN: %clang_cc1 -fmodules -fmodule-map-file=%t/module.modulemap -fmodules-cache-path=%t/cache2 -fdisable-module-hash \
// RUN:   -fsyntax-only %t/tu.c -fmodules-skip-header-search-paths -I %t/two -Rmodule-build 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DID-REUSE --allow-empty
//
// RUN: %clang_cc1 -module-file-info %t/cache2/Mod.pcm | FileCheck %s --check-prefix=NO-HS-PATHS
// NO-HS-PATHS-NOT: Header search paths:
