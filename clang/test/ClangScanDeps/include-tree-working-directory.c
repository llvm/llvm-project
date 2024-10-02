// RUN: rm -rf %t
// RUN: mkdir -p %t/other
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-include-tree-full -cas-path %t/cas \
// RUN:   > %t/deps.json

// Build the include-tree command
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp
// RUN: %clang @%t/tu.rsp -Rcompile-job-cache 2>&1 | FileCheck %s -check-prefix=CACHE-MISS
// RUN: %clang @%t/tu.rsp -Rcompile-job-cache 2>&1 | FileCheck %s -check-prefix=CACHE-HIT
// RUN: ls %t/t.o
// RUN: ls %t/t.d
// RUN: ls %t/t.dia

// CACHE-MISS: remark: compile job cache miss
// CACHE-HIT: remark: compile job cache hit

// RUN: cat %t/tu.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/tu.casid
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/tu.casid | FileCheck %s -DPREFIX=%/t

// CHECK: [[PREFIX]]/t.c llvmcas://
// CHECK: 1:1 <built-in> llvmcas://
// CHECK: 2:1 [[PREFIX]]/relative/h1.h llvmcas://
// CHECK: Files:
// CHECK: [[PREFIX]]/t.c llvmcas://
// CHECK: [[PREFIX]]/relative/h1.h llvmcas://

/// Using a different working directory should cache hit as well.
/// FIXME: Working directory affects some codegen options added by clang driver, preserve them to make sure the cache hit.
// RUN: sed -e "s|DIR|%/t|g" %t/cdb2.json.template > %t/cdb2.json
// RUN: clang-scan-deps -compilation-database %t/cdb2.json -format experimental-include-tree-full -cas-path %t/cas \
// RUN:   > %t/deps2.json
// RUN: %deps-to-rsp %t/deps2.json --tu-index 0 > %t/tu2.rsp
// RUN: %clang @%t/tu2.rsp -Rcompile-job-cache 2>&1 | FileCheck %s -check-prefix=CACHE-HIT


//--- cdb.json.template
[{
  "directory": "DIR/other",
  "command": "clang -c t.c -I relative -working-directory DIR -o t.o -MD -serialize-diagnostics t.dia",
  "file": "DIR/t.c"
}]

//--- cdb2.json.template
[{
  "directory": "DIR/other",
  "command": "clang -c DIR/t.c -I DIR/relative -working-directory DIR/other -o DIR/t2.o -MD -serialize-diagnostics DIR/t2.dia -fdebug-compilation-dir=DIR -fcoverage-compilation-dir=DIR",
  "file": "DIR/t.c"
}]

//--- relative/h1.h

//--- t.c
#include "h1.h"
