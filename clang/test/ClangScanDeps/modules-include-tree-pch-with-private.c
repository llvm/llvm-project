// Test importing a private module whose public module was previously imported
// via a PCH.

// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: sed "s|DIR|%/t|g" %t/cdb_pch.json.template > %t/cdb_pch.json

// RUN: clang-scan-deps -compilation-database %t/cdb_pch.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps_pch.json

// RUN: %deps-to-rsp %t/deps_pch.json --module-name Mod > %t/Mod.rsp
// RUN: %deps-to-rsp %t/deps_pch.json --module-name Indirect1 > %t/Indirect1.rsp
// RUN: %deps-to-rsp %t/deps_pch.json --tu-index 0 > %t/pch.rsp
// RUN: %clang @%t/Mod.rsp
// RUN: %clang @%t/Indirect1.rsp
// RUN: %clang @%t/pch.rsp

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// RUN: %deps-to-rsp %t/deps.json --module-name Mod_Private > %t/Mod_Private.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name Indirect2 > %t/Indirect2.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp
// RUN: %clang @%t/Mod_Private.rsp
// RUN: %clang @%t/Indirect2.rsp
// RUN: %clang @%t/tu.rsp

// Extract include-tree casids
// RUN: cat %t/Indirect2.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/Indirect.casid
// RUN: cat %t/tu.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/tu.casid

// RUN: echo "MODULE Indirect2" > %t/result.txt
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/Indirect.casid >> %t/result.txt
// RUN: echo "TRANSLATION UNIT" >> %t/result.txt
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/tu.casid >> %t/result.txt

// Explicitly check that Mod_Private is imported as a module and not a header.
// RUN: FileCheck %s -DPREFIX=%/t -input-file %t/result.txt

// CHECK-LABEL: MODULE Indirect2
// CHECK: <module-includes> llvmcas://
// CHECK: 1:1 <built-in> llvmcas://
// CHECK: 2:1 [[PREFIX]]/indirect2.h llvmcas://
// CHECK:   Submodule: Indirect2
// CHECK:   2:1 (Module) Indirect1
// CHECK:   3:1 (Module) Mod_Private

// CHECK-LABEL: TRANSLATION UNIT
// CHECK: (PCH) llvmcas://
// CHECK: [[PREFIX]]/tu.m llvmcas://
// CHECK: 1:1 <built-in> llvmcas://
// CHECK: 2:1 (Module) Mod_Private
// CHECK: 3:1 (Module) Indirect2

//--- cdb.json.template
[{
  "file": "DIR/tu.m",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.m -include prefix.h -F DIR -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache"
}]

//--- cdb_pch.json.template
[{
  "file": "DIR/prefix.h",
  "directory": "DIR",
  "command": "clang -x objective-c-header DIR/prefix.h -o DIR/prefix.h.pch -F DIR -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache"
}]

//--- Mod.framework/Modules/module.modulemap
framework module Mod { header "Mod.h" }

//--- Mod.framework/Modules/module.private.modulemap
framework module Mod_Private { header "Priv.h" }

//--- Mod.framework/Headers/Mod.h
void pub(void);

//--- Mod.framework/PrivateHeaders/Priv.h
void priv(void);

//--- module.modulemap
module Indirect1 {
  header "indirect1.h"
  export *
}
module Indirect2 {
  header "indirect2.h"
  export *
}

//--- indirect1.h
#import <Mod/Mod.h>

//--- indirect2.h
#import "indirect1.h"
#import <Mod/Priv.h>

static inline void indirect(void) {
  pub();
  priv();
}

//--- prefix.h
#import <Mod/Mod.h>
#import "indirect1.h"

//--- tu.m
#import <Mod/Priv.h>
#import "indirect2.h"

void tu(void) {
  pub();
  priv();
  indirect();
}
