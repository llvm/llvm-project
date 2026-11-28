// REQUIRES: nvptx-registered-target

static int calleeFn() { return 1; }
int callerFn() { return calleeFn(); }

// Check --plugin-opt=opt-remarks-passes and its --plugin-opt=opt-remarks-filter
// alias.
//
// RUN: %clang -cc1 %s -triple nvptx64-nvidia-cuda -emit-llvm-bc -o %t-remarks.o
//
// RUN: clang-nvlink-wrapper --dry-run %t-remarks.o -arch sm_52 -o a.out \
// RUN:   --plugin-opt=opt-remarks-filename=%t.yaml \
// RUN:   --plugin-opt=opt-remarks-format=yaml \
// RUN:   --plugin-opt=opt-remarks-passes=inline
// RUN: FileCheck %s --check-prefix=REMARKS < %t.yaml
// REMARKS: Pass:{{ +}}inline
//
// RUN: clang-nvlink-wrapper --dry-run %t-remarks.o -arch sm_52 -o a.out \
// RUN:   --plugin-opt=opt-remarks-filename=%t.nomatch.yaml \
// RUN:   --plugin-opt=opt-remarks-format=yaml \
// RUN:   --plugin-opt=opt-remarks-filter=nonexistent-pass-xyz
// RUN: count 0 < %t.nomatch.yaml
