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
// RUN: cat %t/tu.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/tu.casid
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/Top.casid | FileCheck %s -check-prefix=WITH-APINOTES
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/Left.casid | FileCheck %s -check-prefix=WITHOUT-APINOTES
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/tu.casid  | FileCheck %s -check-prefix=WITHOUT-APINOTES

// WITH-APINOTES: APINotes:
// WITH-APINOTES-NEXT: llvmcas://
// WITH-APINOTES-NEXT: Name: Top
// WITH-APINOTES-NEXT: Functions:
// WITH-APINOTES-NEXT:   - Name: top
// WITH-APINOTES-NEXT:     Availability: none
// WITH-APINOTES-NEXT:     AvailabilityMsg: "don't use this"
// WITHOUT-APINOTES-NOT: APINotes:

// Build the include-tree commands
// RUN: %clang @%t/Top.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_MISS
// Ensure the pcm comes from the action cache
// RUN: rm -rf %t/outputs
// RUN: %clang @%t/Left.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_MISS
// RUN: rm -rf %t/outputs
// RUN: %clang @%t/tu.rsp -verify -fno-cache-compile-job

// Check cache hits
// RUN: %clang @%t/Top.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_HIT
// RUN: %clang @%t/Left.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_HIT

// CACHE_MISS: compile job cache miss
// CACHE_HIT: compile job cache hit

// Check incompatible with -fapinotes
// RUN: not %clang -cc1 -fcas-include-tree @%t/Top.casid -fapinotes \
// RUN:   -fcas-path %t/cas -fsyntax-only 2>&1 | \
// RUN:   FileCheck %s -check-prefix=INCOMPATIBLE

// INCOMPATIBLE: error: passing incompatible option '-fapinotes' with '-fcas-include-tree'

//--- cdb.json.template
[{
  "file": "DIR/tu.m",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.m -I DIR -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache -Rcompile-job-cache -fapinotes-modules -iapinotes-modules DIR"
}]

//--- module.modulemap
module Top { header "Top.h" export *}
module Left { header "Left.h" export *}

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

//--- Top.apinotes
Name: Top
Functions:
  - Name: top
    Availability: none
    AvailabilityMsg: "don't use this"
