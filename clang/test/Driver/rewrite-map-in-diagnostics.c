// RUN: export LSAN_OPTIONS=detect_leaks=0
// RUN: rm -rf "%t"
// RUN: mkdir -p "%t"
// RUN: env TMPDIR="%t" TEMP="%t" TMP="%t" RC_DEBUG_OPTION=1 \
// RUN:     not %clang -fsyntax-only -frewrite-map-file=%p/Inputs/rewrite.map %s 2>&1 \
// RUN:   | FileCheck %s

#pragma clang __debug parser_crash

// CHECK: note: diagnostic msg: {{.*}}rewrite.map

// REQUIRES: crash-recovery
