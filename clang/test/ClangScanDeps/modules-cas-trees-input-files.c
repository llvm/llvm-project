// Ensure all files references by a cached module can be found.

// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb_pch.json.template > %t/cdb_pch.json
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// == Scan PCH
// RUN: clang-scan-deps -compilation-database %t/cdb_pch.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps_pch.json

// == Build PCH
// RUN: %deps-to-rsp %t/deps_pch.json --module-name A > %t/A.rsp
// RUN: %deps-to-rsp %t/deps_pch.json --tu-index 0 > %t/pch.rsp

// RUN: %clang @%t/A.rsp
// RUN: %clang @%t/pch.rsp

// == Scan TU, including PCH
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// == Build TU, including PCH
// RUN: %deps-to-rsp %t/deps.json --module-name C > %t/C.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp

// RUN: %clang @%t/C.rsp
// RUN: %clang @%t/tu.rsp

//--- cdb_pch.json.template
[
  {
    "directory" : "DIR",
    "command" : "clang_tool -I DIR -x c-header DIR/prefix.h -o DIR/prefix.h.pch -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache -Rcompile-job-cache",
    "file" : "DIR/prefix.h"
  },
]

//--- cdb.json.template
[
  {
    "directory" : "DIR",
    "command" : "clang_tool -I DIR -fsyntax-only DIR/tu.c -include DIR/prefix.h -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache -Rcompile-job-cache",
    "file" : "DIR/tu.c"
  },
]

// The test below is a specific instance that was failing in the past. The test
// complexity is required to trigger looking up an input file of a module that
// would not be automatically visited when importing that module during
// dependency scanning, because it is only triggered incidentally. The anatomy
// of the specific test case is:
// * module A has a reference to B's modulemap but does not import B
//   because it was for an excluded header
// * module C imports A and performs macro expansion of module_macro
// * looking up the location for the macro definition does binary search and
//   incidentally deserializes the SLocEntry for B's modulemap. With small
//   modules such as in this test case, the binary search is *likely* to hit
//   that specific SLocEntry, and it did at the time this test was written.
// * module A must be "prebuilt"; otherwise the scanner will visit all inputs
//   during implicit module validation; so we load it via PCH.

//--- A/module.modulemap
module A { header "A.h" }

//--- A/A.h
#include "B/B.h"
#define module_macro int

//--- B/module.modulemap
module B { exclude header "B.h" }

//--- B/B.h

//--- module.modulemap
module C { header "C.h" }

//--- C.h
#include "A/A.h"
module_macro x;

//--- prefix.h
#include "A/A.h"

//--- tu.c
#include "C.h"
