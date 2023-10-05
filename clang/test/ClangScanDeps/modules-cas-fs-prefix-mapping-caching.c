// Test that we get cache hits across directories with modules.

// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cp -r %t/dir1 %t/dir2
// RUN: sed -e "s|DIR|%t/dir1|g" -e "s|CLANG|%clang|g" -e "s|SDK|%S/Inputs/SDK|g" %t/cdb.json.template > %t/dir1/cdb.json
// RUN: sed -e "s|DIR|%t/dir2|g" -e "s|CLANG|%clang|g" -e "s|SDK|%S/Inputs/SDK|g" %t/cdb.json.template > %t/dir2/cdb.json

// RUN: clang-scan-deps -compilation-database %t/dir1/cdb.json -format experimental-full \
// RUN:    -cas-path %t/cas -module-files-dir %t/dir1/modules \
// RUN:    -prefix-map-sdk=/^sdk -prefix-map-toolchain=/^tc \
// RUN:    -prefix-map=%t/dir1/modules=/^modules -prefix-map=%t/dir1=/^src \
// RUN:  > %t/dir1.txt

// RUN: clang-scan-deps -compilation-database %t/dir2/cdb.json -format experimental-full \
// RUN:    -cas-path %t/cas -module-files-dir %t/dir2/modules \
// RUN:    -prefix-map-sdk=/^sdk -prefix-map-toolchain=/^tc \
// RUN:    -prefix-map=%t/dir2/modules=/^modules -prefix-map=%t/dir2=/^src \
// RUN:  > %t/dir2.txt

// Extract individual commands.
// RUN: %deps-to-rsp %t/dir1.txt --module-name=_Builtin_stdarg > %t/dir1/stdarg.cc1.rsp
// RUN: %deps-to-rsp %t/dir1.txt --module-name=B > %t/dir1/B.cc1.rsp
// RUN: %deps-to-rsp %t/dir1.txt --module-name=A > %t/dir1/A.cc1.rsp
// RUN: %deps-to-rsp %t/dir1.txt --tu-index 0 > %t/dir1/tu.cc1.rsp

// RUN: %deps-to-rsp %t/dir2.txt --module-name=_Builtin_stdarg > %t/dir2/stdarg.cc1.rsp
// RUN: %deps-to-rsp %t/dir2.txt --module-name=B > %t/dir2/B.cc1.rsp
// RUN: %deps-to-rsp %t/dir2.txt --module-name=A > %t/dir2/A.cc1.rsp
// RUN: %deps-to-rsp %t/dir2.txt --tu-index 0 > %t/dir2/tu.cc1.rsp

// RUN: (cd %t/dir1; %clang @stdarg.cc1.rsp) 2>&1 | FileCheck %s -check-prefix=CACHE-MISS
// RUN: (cd %t/dir1; %clang @B.cc1.rsp) 2>&1 | FileCheck %s -check-prefix=CACHE-MISS
// RUN: (cd %t/dir1; %clang @A.cc1.rsp) 2>&1 | FileCheck %s -check-prefix=CACHE-MISS
// RUN: (cd %t/dir1; %clang @tu.cc1.rsp) 2>&1 | FileCheck %s -check-prefix=CACHE-MISS

// CACHE-MISS: compile job cache miss

// RUN: (cd %t/dir2; %clang @stdarg.cc1.rsp) 2>&1 | FileCheck %s -check-prefix=CACHE-HIT
// RUN: (cd %t/dir2; %clang @B.cc1.rsp) 2>&1 | FileCheck %s -check-prefix=CACHE-HIT
// RUN: (cd %t/dir2; %clang @A.cc1.rsp) 2>&1 | FileCheck %s -check-prefix=CACHE-HIT
// RUN: (cd %t/dir2; %clang @tu.cc1.rsp) 2>&1 | FileCheck %s -check-prefix=CACHE-HIT

// CACHE-HIT: compile job cache hit

// RUN: diff -r -u %t/dir1/modules %t/dir2/modules

//--- cdb.json.template
[
  {
    "directory": "DIR",
    "command": "CLANG -fsyntax-only DIR/t.c -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/mcp -target x86_64-apple-macos11 -isysroot SDK -Rcompile-job-cache",
    "file": "DIR/t.c"
  }
]

//--- dir1/t.c
#include "a.h"

//--- dir1/module.modulemap
module A { header "a.h" }
module B { header "b.h" }

//--- dir1/a.h
#include "b.h"

//--- dir1/b.h
#include <stdarg.h>
#include <stdlib.h>
