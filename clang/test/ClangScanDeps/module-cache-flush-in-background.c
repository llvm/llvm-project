// Test that PCMs built in-process get flushed to disk in the background.

// UNSUPPORTED: system-windows
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// Without flushing, nothing gets written to disk.
// RUN: clang-scan-deps -compilation-database %t/cdb.json -no-flush-module-cache \
// RUN:   -format experimental-full -j 1 -o %t/deps-no-flush.json
// RUN: find %t/cache -name "*.pcm" | count 0

// By default, both PCMs get flushed in the background.
// RUN: rm -rf %t/cache
// RUN: clang-scan-deps -compilation-database %t/cdb.json -log-path=%t/scan.log \
// RUN:   -format experimental-full -j 1 -o %t/deps-flush.json
// RUN: find %t/cache -name "M-*.pcm" | count 1
// RUN: find %t/cache -name "N-*.pcm" | count 1
// RUN: FileCheck %s < %t/scan.log
// RUN: diff %t/deps-no-flush.json %t/deps-flush.json

// CHECK-DAG: pcm_flush: {{.*}}M-{{.*}}.pcm
// CHECK-DAG: pcm_flush: {{.*}}N-{{.*}}.pcm

//--- cdb.json.template
[{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu1.c -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache",
  "file": "DIR/tu1.c"
},{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu2.c -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache",
  "file": "DIR/tu2.c"
}]

//--- module.modulemap
module M { header "M.h" }
module N { header "N.h" }

//--- M.h
void m(void);

//--- N.h
void n(void);

//--- tu1.c
#include "M.h"

//--- tu2.c
#include "N.h"
