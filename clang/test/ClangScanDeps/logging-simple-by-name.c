// Test logging events for the scan-by-name path using CompilerInstanceWithContext.
// This test also covers the case where the compiler spawns a new thread to scan
// an included module.
// Specifically, when the compiler scans dependency for module N on thread TID1,
// it creates a different thread TID2 to scan for M. TID2 discovers that M has
// been built, picks up the up-to-date pcm, and returns back to TID1.
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-full -log-path=%t/scan.log -j 1 \
// RUN:   -module-names=M,N -o %t/deps.json
// RUN: FileCheck %s < %t/scan.log

// CHECK: [{{[0-9]+\.[0-9]+}}] [[#PID:]] [[#TID1:]]: init_compiler_instance_with_context:{{.*}}
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID1]]: start scan_by_name: M
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID1]]: timestamp_read: {{.*}}[[MPCMFILE:.*\.pcm]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID1]]: pcm_write: {{.*}}[[MPCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID1]]: timestamp_write: {{.*}}[[MPCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID1]]: pcm_add_built: {{.*}}[[MPCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID1]]: timestamp_read: {{.*}}[[MPCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID1]]: pcm_read_cached: {{.*}}[[MPCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID1]]: pcm_finalized: {{.*}}[[MPCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID1]]: finish scan_by_name: M
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID1]]: start scan_by_name: N
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID1]]: timestamp_read: {{.*}}[[NPCMFILE:.*\.pcm]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID1]]: pcm_read_cached: {{.*}}[[NPCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID1]]: pcm_read_disk: {{.*}}[[NPCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID2:]]: timestamp_read: {{.*}}[[MPCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID2]]: pcm_read_cached: {{.*}}[[MPCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID2]]: pcm_finalized: {{.*}}[[MPCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID1]]: pcm_write: {{.*}}[[NPCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID1]]: timestamp_write: {{.*}}[[NPCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID1]]: pcm_add_built: {{.*}}[[NPCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID1]]: timestamp_read: {{.*}}[[NPCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID1]]: pcm_read_cached: {{.*}}[[NPCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID1]]: timestamp_read: {{.*}}[[NPCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID1]]: pcm_finalized: {{.*}}[[NPCMFILE]]
// CHECK-NEXT: [{{[0-9]+\.[0-9]+}}] [[#PID]] [[#TID1]]: finish scan_by_name: N

//--- cdb.json.template
[{
  "file": "",
  "directory": "DIR",
  "command": "clang -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -I DIR -fsyntax-only -x c -fbuild-session-timestamp=1 -fmodules-validate-once-per-build-session"
}]

//--- module.modulemap
module M { header "M.h" }
module N { header "N.h" }

//--- M.h
void m(void);

//--- N.h
#include "M.h"
void n(void);