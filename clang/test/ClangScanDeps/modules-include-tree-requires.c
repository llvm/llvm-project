// Check that 'requires' clauses are preserved in include-tree modules.

// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -j 1 \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// Extract the include-tree commands
// RUN: %deps-to-rsp %t/deps.json --module-name Mod > %t/Mod.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name NotCxx > %t/NotCxx.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp

// Print the include trees and check that requirements are preserved.
// RUN: cat %t/Mod.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/Mod.casid
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/Mod.casid | FileCheck %s -check-prefix=MOD
// RUN: cat %t/NotCxx.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/NotCxx.casid
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/NotCxx.casid | FileCheck %s -check-prefix=NOTCXX

// MOD: Module Map:
// MOD-NEXT: Mod
// MOD:        Plain
// MOD:        CxxOnly
// MOD-NEXT:     requires cplusplus

// NOTCXX: Module Map:
// NOTCXX-NEXT: NotCxx
// NOTCXX-NEXT:   requires !cplusplus

// Build the modules and verify compilation succeeds.
// RUN: %clang @%t/Mod.rsp
// RUN: %clang @%t/NotCxx.rsp
// RUN: %clang @%t/tu.rsp

//--- cdb.json.template
[{
  "file": "DIR/tu.c",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.c -I DIR -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache"
}]

//--- module.modulemap
module Mod {
  module Plain { header "Plain.h" }
  module CxxOnly {
    requires cplusplus
    header "CxxOnly.h"
  }
}
module NotCxx {
  requires !cplusplus
  header "NotCxx.h"
}

//--- Plain.h
void plain(void);

//--- CxxOnly.h
void cxx_only(void);

//--- NotCxx.h
void not_cxx(void);

//--- tu.c
#include "Plain.h"
#include "NotCxx.h"

void tu(void) {
  plain();
  not_cxx();
}
