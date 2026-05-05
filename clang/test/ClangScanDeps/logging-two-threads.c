// Test that when two threads are scanning two different TUs, we log an expected
// number of events (26 in this case), and for each pcm file, the sequence
// of events is expected.
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-full -log-path=%t/scan.log -j 2 -o %t/deps.json

// Verify both TUs were scanned.
// RUN: FileCheck %s --check-prefix=TU1 < %t/scan.log
// RUN: FileCheck %s --check-prefix=TU2 < %t/scan.log

// TU1: starting scanning command:{{.*}}tu1.c
// TU1: finished scanning command:{{.*}}tu1.c

// TU2: starting scanning command:{{.*}}tu2.c
// TU2: finished scanning command:{{.*}}tu2.c

// Verify the total number of logged events.
// RUN: wc -l < %t/scan.log | FileCheck %s --check-prefix=LINECOUNT
// LINECOUNT: {{^ *26$}}

// Verify exactly two pcm_writes.
// RUN: grep -c "pcm_write:" %t/scan.log | FileCheck %s --check-prefix=PCMWRITES
// PCMWRITES: 2

// Verify A's pcm is written before B's pcm.
// RUN: grep "pcm_write:" %t/scan.log | FileCheck %s --check-prefix=WRITEORDER
// WRITEORDER: pcm_write: {{.*}}A-{{.*}}.pcm
// WRITEORDER-NEXT: pcm_write: {{.*}}B-{{.*}}.pcm

// Check the correct sequence for building module A.
// RUN: grep "A-" %t/scan.log | FileCheck %s --check-prefix=ASEQ
// ASEQ: timestamp_read: {{.*}}A-{{.*}}.pcm
// ASEQ: pcm_write: {{.*}}A-{{.*}}.pcm
// ASEQ: timestamp_write: {{.*}}A-{{.*}}.pcm
// ASEQ: pcm_add_built: {{.*}}A-{{.*}}.pcm
// ASEQ: pcm_finalized: {{.*}}A-{{.*}}.pcm

// Check the correct sequence for building module B.
// RUN: grep "B-" %t/scan.log | FileCheck %s --check-prefix=BSEQ
// BSEQ: timestamp_read: {{.*}}B-{{.*}}.pcm
// BSEQ: pcm_write: {{.*}}B-{{.*}}.pcm
// BSEQ: timestamp_write: {{.*}}B-{{.*}}.pcm
// BSEQ: pcm_add_built: {{.*}}B-{{.*}}.pcm
// BSEQ: pcm_finalized: {{.*}}B-{{.*}}.pcm

//--- cdb.json.template
[{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu1.c -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -fbuild-session-timestamp=1 -fmodules-validate-once-per-build-session",
  "file": "DIR/tu1.c"
},
{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu2.c -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -fbuild-session-timestamp=1 -fmodules-validate-once-per-build-session",
  "file": "DIR/tu2.c"
}]

//--- module.modulemap
module A { header "A.h" }
module B { header "B.h" }

//--- A.h
void a(void);

//--- B.h
void b(void);

//--- tu1.c
#include "A.h"
void foo(void) { a(); }

//--- tu2.c
#include "A.h"
#include "B.h"
void bar(void) { b(); }