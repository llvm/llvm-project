// Unsupported on AIX because we don't support the requisite "__clangast"
// section in XCOFF yet.
// UNSUPPORTED: aix

// This test checks that the dependency scanner can handle larger amount of
// explicitly built modules retrieved from the PCH.
// (Previously, there was a bug dangling iterator bug that manifested only with
// 16 and more retrieved modules.)

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- mod_00.h
//--- mod_01.h
//--- mod_02.h
//--- mod_03.h
//--- mod_04.h
//--- mod_05.h
//--- mod_06.h
//--- mod_07.h
//--- mod_08.h
//--- mod_09.h
//--- mod_10.h
//--- mod_11.h
//--- mod_12.h
//--- mod_13.h
//--- mod_14.h
//--- mod_15.h
//--- mod_16.h
//--- mod.h
#include "mod_00.h"
#include "mod_01.h"
#include "mod_02.h"
#include "mod_03.h"
#include "mod_04.h"
#include "mod_05.h"
#include "mod_06.h"
#include "mod_07.h"
#include "mod_08.h"
#include "mod_09.h"
#include "mod_10.h"
#include "mod_11.h"
#include "mod_12.h"
#include "mod_13.h"
#include "mod_14.h"
#include "mod_15.h"
#include "mod_16.h"
//--- module.modulemap
module mod_00 { header "mod_00.h" }
module mod_01 { header "mod_01.h" }
module mod_02 { header "mod_02.h" }
module mod_03 { header "mod_03.h" }
module mod_04 { header "mod_04.h" }
module mod_05 { header "mod_05.h" }
module mod_06 { header "mod_06.h" }
module mod_07 { header "mod_07.h" }
module mod_08 { header "mod_08.h" }
module mod_09 { header "mod_09.h" }
module mod_10 { header "mod_10.h" }
module mod_11 { header "mod_11.h" }
module mod_12 { header "mod_12.h" }
module mod_13 { header "mod_13.h" }
module mod_14 { header "mod_14.h" }
module mod_15 { header "mod_15.h" }
module mod_16 { header "mod_16.h" }
module mod    { header "mod.h"    }

//--- pch.h
#include "mod.h"

//--- tu.c

//--- cdb_pch.json.template
[{
  "file": "DIR/pch.h",
  "directory": "DIR",
  "command": "clang -x c-header DIR/pch.h -fmodules -gmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -o DIR/pch.h.gch"
}]

//--- cdb_tu.json.template
[{
  "file": "DIR/tu.c",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.c -fmodules -gmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -include DIR/pch.h -o DIR/tu.o"
}]

// Scan dependencies of the PCH:
//
// RUN: sed "s|DIR|%/t|g" %t/cdb_pch.json.template > %t/cdb_pch.json
// RUN: clang-scan-deps -compilation-database %t/cdb_pch.json -format experimental-full \
// RUN:   -generate-modules-path-args -module-files-dir %t/build > %t/result_pch.json

// Explicitly build the PCH:
//
// RUN: %deps-to-rsp %t/result_pch.json --module-name=mod_00 > %t/mod_00.cc1.rsp
// RUN: %deps-to-rsp %t/result_pch.json --module-name=mod_01 > %t/mod_01.cc1.rsp
// RUN: %deps-to-rsp %t/result_pch.json --module-name=mod_02 > %t/mod_02.cc1.rsp
// RUN: %deps-to-rsp %t/result_pch.json --module-name=mod_03 > %t/mod_03.cc1.rsp
// RUN: %deps-to-rsp %t/result_pch.json --module-name=mod_04 > %t/mod_04.cc1.rsp
// RUN: %deps-to-rsp %t/result_pch.json --module-name=mod_05 > %t/mod_05.cc1.rsp
// RUN: %deps-to-rsp %t/result_pch.json --module-name=mod_06 > %t/mod_06.cc1.rsp
// RUN: %deps-to-rsp %t/result_pch.json --module-name=mod_07 > %t/mod_07.cc1.rsp
// RUN: %deps-to-rsp %t/result_pch.json --module-name=mod_08 > %t/mod_08.cc1.rsp
// RUN: %deps-to-rsp %t/result_pch.json --module-name=mod_09 > %t/mod_09.cc1.rsp
// RUN: %deps-to-rsp %t/result_pch.json --module-name=mod_10 > %t/mod_10.cc1.rsp
// RUN: %deps-to-rsp %t/result_pch.json --module-name=mod_11 > %t/mod_11.cc1.rsp
// RUN: %deps-to-rsp %t/result_pch.json --module-name=mod_12 > %t/mod_12.cc1.rsp
// RUN: %deps-to-rsp %t/result_pch.json --module-name=mod_13 > %t/mod_13.cc1.rsp
// RUN: %deps-to-rsp %t/result_pch.json --module-name=mod_14 > %t/mod_14.cc1.rsp
// RUN: %deps-to-rsp %t/result_pch.json --module-name=mod_15 > %t/mod_15.cc1.rsp
// RUN: %deps-to-rsp %t/result_pch.json --module-name=mod_16 > %t/mod_16.cc1.rsp
// RUN: %deps-to-rsp %t/result_pch.json --module-name=mod    > %t/mod.cc1.rsp
// RUN: %deps-to-rsp %t/result_pch.json --tu-index=0 > %t/pch.rsp
//
// RUN: %clang @%t/mod_00.cc1.rsp
// RUN: %clang @%t/mod_01.cc1.rsp
// RUN: %clang @%t/mod_02.cc1.rsp
// RUN: %clang @%t/mod_03.cc1.rsp
// RUN: %clang @%t/mod_04.cc1.rsp
// RUN: %clang @%t/mod_05.cc1.rsp
// RUN: %clang @%t/mod_06.cc1.rsp
// RUN: %clang @%t/mod_07.cc1.rsp
// RUN: %clang @%t/mod_08.cc1.rsp
// RUN: %clang @%t/mod_09.cc1.rsp
// RUN: %clang @%t/mod_10.cc1.rsp
// RUN: %clang @%t/mod_11.cc1.rsp
// RUN: %clang @%t/mod_12.cc1.rsp
// RUN: %clang @%t/mod_13.cc1.rsp
// RUN: %clang @%t/mod_14.cc1.rsp
// RUN: %clang @%t/mod_15.cc1.rsp
// RUN: %clang @%t/mod_16.cc1.rsp
// RUN: %clang @%t/mod.cc1.rsp
// RUN: %clang @%t/pch.rsp

// Scan dependencies of the TU, checking it doesn't crash:
//
// RUN: sed "s|DIR|%/t|g" %t/cdb_tu.json.template > %t/cdb_tu.json
// RUN: clang-scan-deps -compilation-database %t/cdb_tu.json -format experimental-full \
// RUN:   -generate-modules-path-args -module-files-dir %t/build
