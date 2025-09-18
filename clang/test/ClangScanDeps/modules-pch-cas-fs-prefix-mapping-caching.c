// Test that we get cache hits across directories with modules and PCH.

// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cp -r %t/dir1 %t/dir2
// RUN: sed -e "s|DIR|%/t/dir1|g" -e "s|CLANG|%/ncclang|g" -e "s|SDK|%/S/Inputs/SDK|g" %t/cdb.json.template > %t/dir1/cdb.json
// RUN: sed -e "s|DIR|%/t/dir1|g" -e "s|CLANG|%/ncclang|g" -e "s|SDK|%/S/Inputs/SDK|g" %t/cdb_pch.json.template > %t/dir1/cdb_pch.json
// RUN: sed -e "s|DIR|%/t/dir2|g" -e "s|CLANG|%/ncclang|g" -e "s|SDK|%/S/Inputs/SDK|g" %t/cdb.json.template > %t/dir2/cdb.json
// RUN: sed -e "s|DIR|%/t/dir2|g" -e "s|CLANG|%/ncclang|g" -e "s|SDK|%/S/Inputs/SDK|g" %t/cdb_pch.json.template > %t/dir2/cdb_pch.json

// == Scan PCH
// RUN: clang-scan-deps -compilation-database %t/dir1/cdb_pch.json -format experimental-full -optimize-args=none \
// RUN:    -cas-path %t/cas -module-files-dir %t/dir1/modules \
// RUN:    -prefix-map=%t/dir1/modules=%/root^modules -prefix-map=%t/dir1=%/root^src -prefix-map-sdk=%/root^sdk -prefix-map-toolchain=%/root^tc \
// RUN:  > %t/pch_dir1.txt

// RUN: clang-scan-deps -compilation-database %t/dir2/cdb_pch.json -format experimental-full -optimize-args=none \
// RUN:    -cas-path %t/cas -module-files-dir %t/dir2/modules \
// RUN:    -prefix-map=%t/dir2/modules=%/root^modules -prefix-map=%t/dir2=%/root^src -prefix-map-sdk=%/root^sdk -prefix-map-toolchain=%/root^tc \
// RUN:  > %t/pch_dir2.txt

// == Build PCH
// RUN: %deps-to-rsp %t/pch_dir1.txt --module-name=B > %t/dir1/B.cc1.rsp
// RUN: %deps-to-rsp %t/pch_dir1.txt --module-name=A > %t/dir1/A.cc1.rsp
// RUN: %deps-to-rsp %t/pch_dir1.txt --tu-index 0 > %t/dir1/pch.cc1.rsp
// RUN: cd %t/dir1 && %clang @B.cc1.rsp > %t/miss-B.txt 2>&1
// RUN: cat %t/miss-B.txt | FileCheck %s -check-prefix=CACHE-MISS
// RUN: cd %t/dir1 && %clang @A.cc1.rsp > %t/miss-A.txt 2>&1
// RUN: cat %t/miss-A.txt | FileCheck %s -check-prefix=CACHE-MISS
// RUN: cd %t/dir1 && %clang @pch.cc1.rsp > %t/miss-pch.txt 2>&1
// RUN: cat %t/miss-pch.txt | FileCheck %s -check-prefix=CACHE-MISS

// CACHE-MISS: compile job cache miss

// RUN: %deps-to-rsp %t/pch_dir2.txt --module-name=B > %t/dir2/B.cc1.rsp
// RUN: %deps-to-rsp %t/pch_dir2.txt --module-name=A > %t/dir2/A.cc1.rsp
// RUN: %deps-to-rsp %t/pch_dir2.txt --tu-index 0 > %t/dir2/pch.cc1.rsp
// RUN: cd %t/dir2 && %clang @B.cc1.rsp > %t/hit-B.txt 2>&1
// RUN: cat %t/hit-B.txt | FileCheck %s -check-prefix=CACHE-HIT
// RUN: cd %t/dir2 && %clang @A.cc1.rsp > %t/hit-A.txt 2>&1
// RUN: cat %t/hit-B.txt | FileCheck %s -check-prefix=CACHE-HIT
// RUN: cd %t/dir2 && %clang @pch.cc1.rsp > %t/hit-pch.txt 2>&1
// RUN: cat %t/hit-pch.txt | FileCheck %s -check-prefix=CACHE-HIT

// CACHE-HIT: compile job cache hit

// == Scan TU, including PCH
// RUN: clang-scan-deps -compilation-database %t/dir1/cdb.json -format experimental-full -optimize-args=none \
// RUN:    -cas-path %t/cas -module-files-dir %t/dir1/modules \
// RUN:    -prefix-map=%t/dir1/modules=%/root^modules -prefix-map=%t/dir1=%/root^src -prefix-map-sdk=%/root^sdk -prefix-map-toolchain=%/root^tc \
// RUN:  > %t/dir1.txt

// RUN: clang-scan-deps -compilation-database %t/dir2/cdb.json -format experimental-full -optimize-args=none \
// RUN:    -cas-path %t/cas -module-files-dir %t/dir2/modules \
// RUN:    -prefix-map=%t/dir2/modules=%/root^modules -prefix-map=%t/dir2=%/root^src -prefix-map-sdk=%/root^sdk -prefix-map-toolchain=%/root^tc \
// RUN:  > %t/dir2.txt

// == Build TU
// RUN: %deps-to-rsp %t/dir1.txt --module-name=C > %t/dir1/C.cc1.rsp
// RUN: %deps-to-rsp %t/dir1.txt --tu-index 0 > %t/dir1/tu.cc1.rsp
// RUN: cd %t/dir1 && %clang @C.cc1.rsp > %t/c-miss.txt 2>&1
// RUN: cat %t/c-miss.txt | FileCheck %s -check-prefix=CACHE-MISS
// RUN: cd %t/dir1 && %clang @tu.cc1.rsp > %t/tu-miss.txt 2>&1
// RUN: cat %t/tu-miss.txt | FileCheck %s -check-prefix=CACHE-MISS

// RUN: %deps-to-rsp %t/dir2.txt --module-name=C > %t/dir2/C.cc1.rsp
// RUN: %deps-to-rsp %t/dir2.txt --tu-index 0 > %t/dir2/tu.cc1.rsp
// RUN: cd %t/dir2 && %clang @C.cc1.rsp > %t/c-hit.txt 2>&1
// RUN: cat %t/c-hit.txt | FileCheck %s -check-prefix=CACHE-HIT
// RUN: cd %t/dir2 && %clang @tu.cc1.rsp > %t/tu-hit.txt  2>&1
// RUN: cat %t/tu-hit.txt | FileCheck %s -check-prefix=CACHE-HIT

// RUN: diff -u %t/dir1/prefix.h.pch %t/dir2/prefix.h.pch
// RUN: diff -r -u %t/dir1/modules %t/dir2/modules

//--- cdb.json.template
[
  {
    "directory": "DIR",
    "command": "CLANG -fsyntax-only DIR/t.c -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/mcp -target x86_64-apple-macos11 -isysroot SDK -include DIR/prefix.h -Rcompile-job-cache",
    "file": "DIR/t.c"
  }
]

//--- cdb_pch.json.template
[
  {
    "directory" : "DIR",
    "command" : "CLANG -x c-header DIR/prefix.h -o DIR/prefix.h.pch -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/mcp -target x86_64-apple-macos11 -isysroot SDK -Rcompile-job-cache",
    "file" : "DIR/prefix.h"
  },
]

//--- dir1/t.c
#include "c.h"

//--- dir1/prefix.h
#include "a.h"

//--- dir1/module.modulemap
module A { header "a.h" }
module B { header "b.h" }
module C { header "c.h" }

//--- dir1/a.h
#include "b.h"

//--- dir1/b.h
#include <stdarg.h>
#include <stdlib.h>

//--- dir1/c.h
#include "b.h"
