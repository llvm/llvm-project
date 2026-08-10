// RUN: rm -f %t.cp.c
// RUN: cp %s %t.cp.c
// RUN: clang-refactor local-rename -selection=%t.cp.c:6:5 -new-name=test -v %t.cp.c -- | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor local-rename -selection=%t.cp.c:6:5-6:9 -new-name=test -v %t.cp.c -- | FileCheck --check-prefix=CHECK2 %s

int test;

// CHECK1: invoking action 'local-rename':
// CHECK1-NEXT: -selection={{.*}}.cp.c:6:5 -> {{.*}}.cp.c:6:5

// CHECK2: invoking action 'local-rename':
// CHECK2-NEXT: -selection={{.*}}.cp.c:6:5 -> {{.*}}.cp.c:6:9

// RUN: not clang-refactor local-rename -selection=%s:6:5 -new-name=test -v %t.cp.c -- 2>&1 | FileCheck --check-prefix=CHECK-FILE-ERR %s
// CHECK-FILE-ERR: given file is not in the target TU
