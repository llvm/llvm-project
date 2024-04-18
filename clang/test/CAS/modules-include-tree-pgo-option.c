// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: llvm-profdata merge -o %t/instrumentation.profdata %S/Inputs/pgo.profraw
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// RUN: %deps-to-rsp %t/deps.json --module-name Top > %t/Top.rsp
// RUN: FileCheck %s --input-file=%t/Top.rsp

// RUN: cat %t/Top.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/Top.casid
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/Top.casid | FileCheck %s
// CHECK-NOT: instrumentation.profdata

//--- cdb.json.template
[{
  "file": "DIR/tu.m",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.m -I DIR -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache -Rcompile-job-cache -fprofile-instr-use=DIR/instrumentation.profdata"
}]

//--- module.modulemap
module Top { header "Top.h" export *}

//--- Top.h
#pragma once
struct Top {
  int x;
};
void top(void);

//--- tu.m
#import "Top.h"

void tu(void) {
  top();
}
