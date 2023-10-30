// This test checks that __has_include(<FW/PrivateHeader.h>) in a module does
// not clobber #include <FW/PrivateHeader.h> in importers of said module.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- cdb.json.template
[{
  "file": "DIR/tu.c",
  "directory": "DIR",
  "command": "clang DIR/tu.c -fmodules -fmodules-cache-path=DIR/cache -I DIR/modules -F DIR/frameworks -o DIR/tu.o"
}]

//--- frameworks/FW.framework/Modules/module.private.modulemap
framework module FW_Private {
  umbrella header "A.h"
  module * { export * }
}
//--- frameworks/FW.framework/PrivateHeaders/A.h
#include <FW/B.h>
//--- frameworks/FW.framework/PrivateHeaders/B.h
#include "dependency.h"

//--- modules/module.modulemap
module Poison { header "poison.h" }
module Import { header "import.h" }
module Dependency { header "dependency.h" }
//--- modules/poison.h
#if __has_include(<FW/B.h>)
#define HAS_B 1
#else
#define HAS_B 0
#endif
//--- modules/import.h
#include <FW/B.h>
//--- modules/dependency.h

//--- tu.c
#include "poison.h"

#if __has_include(<FW/B.h>)
#endif

#include "import.h"

#include <FW/B.h>

// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-include-tree-full -cas-path %t/cas > %t/deps.json
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp
// RUN: cat %t/tu.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/tu.casid
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/tu.casid | FileCheck %s -DPREFIX=%/t

// Let's check that the TU actually imports FW_Private.B instead of treating FW/B.h as textual.
// CHECK:      [[PREFIX]]/tu.c llvmcas://
// CHECK-NEXT: 1:1 <built-in> llvmcas://
// CHECK-NEXT: 2:1 (Module) Poison
// CHECK-NEXT: 7:1 (Module) Import
// CHECK-NEXT: 9:1 (Module) FW_Private.B
