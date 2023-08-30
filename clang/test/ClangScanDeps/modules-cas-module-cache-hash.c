// Check that using the same module cache does not cause errors when switching
// between cas-fs and include-tree.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-include-tree-full | FileCheck %s -check-prefix=INCLUDE_TREE

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-full | FileCheck %s  -check-prefix=CAS_FS

// INCLUDE_TREE: "-fcas-include-tree"
// CAS_FS: "-fcas-fs"

//--- cdb.json.template
[{
  "file": "DIR/tu.c",
  "directory": "DIR",
  "command": "clang -fsyntax-only -fmodules -fimplicit-modules -fmodules-cache-path=DIR/mcp DIR/tu.c"
}]

//--- module.modulemap
module M { header "M.h" }

//--- M.h

//--- tu.c
#include "M.h"
