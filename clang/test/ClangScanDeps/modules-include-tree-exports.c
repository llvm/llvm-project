// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -j 1 \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// Extract the include-tree commands
// RUN: %deps-to-rsp %t/deps.json --module-name TwoSubs > %t/TwoSubs.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name ExportExplicit > %t/ExportExplicit.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name ExportWildcard > %t/ExportWildcard.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name ExportGlobalWildcard > %t/ExportGlobalWildcard.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name NoExports > %t/NoExports.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu_export_explicit.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 1 > %t/tu_export_global_wildcard.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 2 > %t/tu_export_none.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 3 > %t/tu_export_wildcard.rsp

// Build
// RUN: %clang @%t/TwoSubs.rsp
// RUN: %clang @%t/ExportExplicit.rsp
// RUN: %clang @%t/ExportWildcard.rsp
// RUN: %clang @%t/ExportGlobalWildcard.rsp
// RUN: %clang @%t/NoExports.rsp
// RUN: not %clang @%t/tu_export_explicit.rsp 2>&1 | FileCheck %s -check-prefix=tu_export_explicit
// RUN: %clang @%t/tu_export_wildcard.rsp 2>&1 | FileCheck %s -check-prefix=tu_export_wildcard -allow-empty
// RUN: %clang @%t/tu_export_global_wildcard.rsp 2>&1 | FileCheck %s -check-prefix=tu_export_global_wildcard -allow-empty
// RUN: not %clang @%t/tu_export_none.rsp 2>&1 | FileCheck %s -check-prefix=tu_export_none

//--- cdb.json.template
[
{
  "file": "DIR/tu_export_explicit.c",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu_export_explicit.c -I DIR -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache"
},
{
  "file": "DIR/tu_export_wildcard.c",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu_export_wildcard.c -I DIR -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache"
},
{
  "file": "DIR/tu_export_global_wildcard.c",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu_export_global_wildcard.c -I DIR -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache"
},
{
  "file": "DIR/tu_export_none.c",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu_export_none.c -I DIR -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache"
},
]

//--- module.modulemap
module TwoSubs {
  module Sub1 { header "Sub1.h" }
  module Sub2 { header "Sub2.h" }
}

module ExportExplicit {
  header "Import.h"
  export TwoSubs.Sub2
}

module ExportWildcard {
  header "Import.h"
  export TwoSubs.*
}

module ExportGlobalWildcard {
  header "Import.h"
  export *
}

module NoExports {
  header "Import.h"
}

//--- Sub1.h
void sub1(void);

//--- Sub2.h
void sub2(void);

//--- Import.h
#include "Sub1.h"
#include "Sub2.h"

//--- tu_export_explicit.c
#pragma clang module import ExportExplicit
void tu1(void) {
  sub2();
  // tu_export_explicit-NOT: error
  sub1();
  // tu_export_explicit: error: call to undeclared function 'sub1'
  // tu_export_explicit: error: missing '#include "Sub1.h"'
}

//--- tu_export_wildcard.c
#pragma clang module import ExportWildcard
void tu1(void) {
  sub1();
  sub2();
  // tu_export_wildcard-NOT: error
}

//--- tu_export_global_wildcard.c
#pragma clang module import ExportGlobalWildcard
void tu1(void) {
  sub1();
  sub2();
  // tu_export_global_wildcard-NOT: error
}

//--- tu_export_none.c
#pragma clang module import NoExports
void tu1(void) {
  sub1();
  // tu_export_none: error: call to undeclared function 'sub1'
  // tu_export_none: error: missing '#include "Sub1.h"'
  sub2();
  // tu_export_none: error: call to undeclared function 'sub2'
  // tu_export_none: error: missing '#include "Sub2.h"'
}
