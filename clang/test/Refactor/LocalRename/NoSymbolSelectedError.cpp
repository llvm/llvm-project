// RUN: not clang-refactor local-rename -selection=%s:4:1 -new-name=Bar %s -- 2>&1 | FileCheck %s
// RUN: clang-refactor local-rename -selection=test:%s -new-name=Bar %s -- 2>&1 | FileCheck --check-prefix=TESTCHECK %s

class Baz { // CHECK: [[@LINE]]:1: error: there is no symbol at the given location
};
/*range=*/;
// TESTCHECK:      1 '' results:
// TESTCHECK-NEXT: there is no symbol at the given location
