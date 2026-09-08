// Test "pcm_dropped" event logging.
// The test performs the same dependency scanning twice, with a change in the
// the header content in between.

// UNSUPPORTED: system-windows
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-full -log-path=%t/scan1.log -j 1 -o %t/deps1.json

// RUN: echo "void a2(void);" >> %t/A.h

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-full -log-path=%t/scan2.log -j 1 -o %t/deps2.json
// RUN: FileCheck %s < %t/scan2.log

// The stale PCM is loaded as tentative (pcm_add) and dropped (pcm_dropped) on
// the same thread.
// CHECK:      [{{[0-9]+\.[0-9]+}}] [[#PID:]] [[#TID:]]: pcm_add: {{.*}}[[PCM:A-.*\.pcm]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID]]: pcm_dropped: {{.*}}[[PCM]]
// CHECK:      pcm_add_built: {{.*}}[[PCM]]
// CHECK:      pcm_finalized: {{.*}}[[PCM]]

//--- cdb.json.template
[{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.c -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache",
  "file": "DIR/tu.c"
}]

//--- module.modulemap
module A { header "A.h" }

//--- A.h
void a(void);

//--- tu.c
#include "A.h"
void foo(void) { a(); }
