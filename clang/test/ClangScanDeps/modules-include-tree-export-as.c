// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// Extract the include-tree commands
// RUN: %deps-to-rsp %t/deps.json --module-name Top > %t/Top.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name Left > %t/Left.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp

// Extract include-tree casids
// RUN: cat %t/Top.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/Top.casid
// RUN: cat %t/Left.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/Left.casid
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/Top.casid | FileCheck %s -check-prefix=TOP
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/Left.casid | FileCheck %s -check-prefix=LEFT

// TOP: Module Map:
// TOP-NEXT: Top
// TOP-NEXT: export_as Left
// LEFT: Module Map:
// LEFT-NEXT Left
// LEFT-NOT: export_as

//--- cdb.json.template
[{
  "file": "DIR/tu.m",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.m -I DIR -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache -Rcompile-job-cache -fapinotes-modules -iapinotes-modules DIR"
}]

//--- module.modulemap
module Top {
  header "Top.h"
  export_as Left
  export *
}
module Left {
  header "Left.h"
  export *
}

//--- Top.h
#pragma once
struct Top {
  int x;
};
void top(void);

//--- Left.h
#include "Top.h"
void left(void);

//--- tu.m
#import "Left.h"

void tu(void) {
  top(); // expected-error {{'top' is unavailable: don't use this}}
  left();
}
// expected-note@Top.h:5{{'top' has been explicitly marked unavailable here}}

