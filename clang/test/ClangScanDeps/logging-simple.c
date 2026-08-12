// Test the strict sequence of events when building a single scanning pcm.

// UNSUPPORTED: system-windows
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-full -log-path=%t/scan.log -j 1 -o %t/deps.json
// RUN: FileCheck %s < %t/scan.log

// The check lines below form a strict sequence of events logged when we only
// build a single scanning pcm. We should only log these events, no more and no
// less, strictly in this order. Changes to this list should be intentional.

// CHECK: [{{[0-9]+\.[0-9]+}}] [[#PID:]] [[#TID:]]: starting scanning command:{{.*}}tu.c
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID]]: init_compiler_instance_with_context:{{.*}}
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID]]: timestamp_read: {{.*}}[[PCMFILE:.*\.pcm]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID]]: pcm_read_cached: {{.*}}[[PCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID]]: pcm_read_disk: {{.*}}[[PCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID2:]]: module_compile_thread: parent=[[#TID]] pcm_compile: {{.*}}[[PCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID]]: pcm_write: {{.*}}[[PCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID]]: timestamp_write: {{.*}}[[PCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID]]: pcm_add_built: {{.*}}[[PCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID]]: timestamp_read: {{.*}}[[PCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID]]: pcm_read_cached: {{.*}}[[PCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID]]: pcm_finalized: {{.*}}[[PCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID]]: finished scanning command:{{.*}}tu.c

//--- cdb.json.template
[{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.c -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -fbuild-session-timestamp=1 -fmodules-validate-once-per-build-session",
  "file": "DIR/tu.c"
}]

//--- module.modulemap
module M { header "M.h" }

//--- M.h
void m(void);

//--- tu.c
#include "M.h"
void foo(void) { m(); }
