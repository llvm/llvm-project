// Verify that dependency scanning produces identical results with and without
// -async-scan-modules so we exercise the async scan code path.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -j 1 \
// RUN:   -format experimental-full -mode preprocess-dependency-directives \
// RUN:   > %t/sync.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -j 1 -async-scan-modules \
// RUN:   -format experimental-full -mode preprocess-dependency-directives \
// RUN:   > %t/async.json

// RUN: diff -u %t/sync.json %t/async.json

// Check that the scan computed the correct module graph.
// RUN: FileCheck %s < %t/sync.json
// CHECK-DAG: "name": "A"
// CHECK-DAG: "name": "B"
// CHECK-DAG: "name": "C"

//--- cdb.json.in

[{
  "directory": "DIR",
  "command": "clang -c DIR/main.c -IDIR -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-modules -fimplicit-module-maps",
  "file": "DIR/main.c"
}]

//--- module.modulemap

module A {
  header "a.h"
}

module B {
  header "b.h"
}

module C {
  header "c.h"
}

//--- a.h

#include "b.h"
void a(void);

//--- b.h

void b(void);

//--- c.h

void c(void);

//--- main.c

#include "a.h"
#include "c.h"
void m(void) {
  a();
  c();
}
