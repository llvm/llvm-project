// UNSUPPORTED: target={{.*}}-aix{{.*}}

/// Test that a signature mismatch on a PCH's module dependency is diagnosed
/// during dependency scanning.
///
/// 1. Scan PCH dependencies with modules hash content encoded.
/// 2. Explicitly build module A and the PCH.
/// 3. Modify A's header, resulting in the same byte length and rebuild A for a new signature.
/// 4. Scan a TU that uses the out of date PCH to detect signature mismatch.

// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

// RUN: sed "s|DIR|%/t|g" %t/cdb_pch.json.in > %t/cdb_pch.json
// RUN: clang-scan-deps -compilation-database %t/cdb_pch.json \
// RUN:   -format experimental-full -module-files-dir %t/build > %t/result_pch.json

// RUN: %deps-to-rsp %t/result_pch.json --module-name=A > %t/A.rsp
// RUN: %deps-to-rsp %t/result_pch.json --tu-index=0 > %t/pch.rsp
// RUN: %clang @%t/A.rsp
// RUN: %clang @%t/pch.rsp

// RUN: cp %t/a_alt.h %t/a.h
// RUN: %clang @%t/A.rsp

// RUN: sed "s|DIR|%/t|g" %t/cdb_tu.json.in > %t/cdb_tu.json
// RUN: not clang-scan-deps -compilation-database %t/cdb_tu.json \
// RUN:   -format experimental-full -module-files-dir %t/build 2>&1 \
// RUN:   | FileCheck %s

// CHECK: error:{{.*}}is out of date and needs to be rebuilt
// CHECK: note: earlier input file validation was disabled for this kind of precompiled file
// CHECK: note: unable to verify precompiled file signature: signature mismatch

//--- module.modulemap
module A {
  header "a.h"
}

//--- a.h
typedef int A_t;

//--- a_alt.h
typedef int A_u;

//--- pch.h
#include "a.h"

//--- tu.c

//--- cdb_pch.json.in
[{
  "directory": "DIR",
  "command": "clang -x c-header DIR/pch.h -fmodules -fmodules-cache-path=DIR/cache -Xclang -fmodules-hash-content -I DIR -o DIR/pch.h.pch",
  "file": "DIR/pch.h"
}]

//--- cdb_tu.json.in
[{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.c -fmodules -fmodules-cache-path=DIR/cache -Xclang -fmodules-hash-content -I DIR -include DIR/pch.h -o DIR/tu.o",
  "file": "DIR/tu.c"
}]
