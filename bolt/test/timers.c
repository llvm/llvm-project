/* This test checks timers for metadata manager phases.
# RUN: %clang %cflags %s -o %t.exe
# RUN: link_fdata %s %t.exe %t.fdata
# RUN: llvm-bolt %t.exe -o %t.null --data %t.fdata -w %t.yaml --time-rewrite \
# RUN:   | FileCheck %s
# RUN: link_fdata %s %t.exe %t.preagg PREAGG
# RUN: perf2bolt %t.exe -o %t.null -p %t.preagg --pa --time-rewrite \
# RUN:   | FileCheck %s

# CHECK-DAG: update metadata post-emit
# CHECK-DAG: process section metadata
# CHECK-DAG: process metadata pre-CFG
# CHECK-DAG: process metadata post-CFG
# CHECK-DAG: finalize metadata pre-emit

# FDATA: 0 [unknown] 0 1 main 0 1 0
# PREAGG: B X:0 #main# 1 0
*/
int main() { return 0; }
