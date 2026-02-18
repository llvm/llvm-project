// Check that the scanner produces raw ast files, even when builds produce the
// obj format, and that the scanner can read obj format from PCH and modules
// imported by PCH.

// Unsupported on AIX because we don't support the requisite "__clangast"
// section in XCOFF yet.
// UNSUPPORTED: target={{.*}}-aix{{.*}}

// RUN: rm -rf %t && mkdir %t
// RUN: cp %S/Inputs/modules-pch/* %t

// Scan dependencies of the PCH:
//
// RUN: rm -f %t/cdb_pch.json
// RUN: sed "s|DIR|%/t|g" %S/Inputs/modules-pch/cdb_pch.json > %t/cdb_pch.json
// RUN: clang-scan-deps -compilation-database %t/cdb_pch.json -format experimental-full \
// RUN:   -module-files-dir %t/build -o %t/result_pch.json

// Explicitly build the PCH:
//
// RUN: %deps-to-rsp %t/result_pch.json --module-name=ModCommon1 > %t/mod_common_1.cc1.rsp
// RUN: %deps-to-rsp %t/result_pch.json --module-name=ModCommon2 > %t/mod_common_2.cc1.rsp
// RUN: %deps-to-rsp %t/result_pch.json --module-name=ModPCH > %t/mod_pch.cc1.rsp
// RUN: %deps-to-rsp %t/result_pch.json --tu-index=0 > %t/pch.rsp
//
// RUN: %clang @%t/mod_common_1.cc1.rsp
// RUN: %clang @%t/mod_common_2.cc1.rsp
// RUN: %clang @%t/mod_pch.cc1.rsp
// RUN: %clang @%t/pch.rsp

// Scan dependencies of the TU:
//
// RUN: rm -f %t/cdb_tu.json
// RUN: sed "s|DIR|%/t|g" %S/Inputs/modules-pch/cdb_tu.json > %t/cdb_tu.json
// RUN: clang-scan-deps -compilation-database %t/cdb_tu.json -format experimental-full \
// RUN:   -module-files-dir %t/build > %t/result_tu.json

// Explicitly build the TU:
//
// RUN: %deps-to-rsp %t/result_tu.json --module-name=ModTU > %t/mod_tu.cc1.rsp
// RUN: %deps-to-rsp %t/result_tu.json --tu-index=0 > %t/tu.rsp
//
// RUN: %clang @%t/mod_tu.cc1.rsp
// RUN: %clang @%t/tu.rsp

// Check the module format for scanner modules:
//
// RUN: find %t/cache -name "*.pcm" -exec %clang_cc1 -module-file-info "{}" ";" | FileCheck %s -check-prefix=SCAN
// SCAN: Module format: raw
// SCAN: Module format: raw
// SCAN: Module format: raw
// SCAN: Module format: raw

// Check the module format for built modules:
//
// RUN: find %t/build -name "*.pcm" -exec %clang_cc1 -module-file-info "{}" ";" | FileCheck %s -check-prefix=BUILD
// BUILD: Module format: obj
// BUILD: Module format: obj
// BUILD: Module format: obj
// BUILD: Module format: obj

// FIXME: check pch format as well; -module-file-info does not work with a PCH
