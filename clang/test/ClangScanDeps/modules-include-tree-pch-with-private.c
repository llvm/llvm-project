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
// RUN: %deps-to-rsp %t/deps_pch.json --tu-index 0 > %t/pch.rsp
// RUN: %clang @%t/Mod.rsp
// RUN: %clang @%t/pch.rsp

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// RUN: %deps-to-rsp %t/deps.json --module-name Mod_Private > %t/Mod_Private.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp
// RUN: %clang @%t/Mod_Private.rsp
// RUN: %clang @%t/tu.rsp

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

//--- prefix.h
#import <Mod/Mod.h>

//--- tu.m
#import <Mod/Priv.h>
void tu(void) {
  pub();
  priv();
}
