// RUN: clang-refactor local-rename -selection=test:%s -new-name=test -v %s -- | FileCheck %s

/*range=*/int test;

/*range named=*/int test2;

/*range= +1*/int test3;

/* range = +100 */int test4;

/*range named =+0*/int test5;

/*range =->+0:22*/int test6;

// CHECK: Test selection group '':
// CHECK-NEXT:   105-105
// CHECK-NEXT:   158-158
// CHECK-NEXT:   197-197
// CHECK-NEXT:   248-251
// CHECK-NEXT: Test selection group 'named':
// CHECK-NEXT:   132-132
// CHECK-NEXT:   218-218

// The following invocations are in the default group:

// CHECK: invoking action 'local-rename':
// CHECK-NEXT: -selection={{.*}}tool-test-support.c:3:11

// CHECK: invoking action 'local-rename':
// CHECK-NEXT: -selection={{.*}}tool-test-support.c:7:15

// CHECK: invoking action 'local-rename':
// CHECK-NEXT: -selection={{.*}}tool-test-support.c:9:29

// CHECK: invoking action 'local-rename':
// CHECK-NEXT: -selection={{.*}}tool-test-support.c:13:19 -> {{.*}}tool-test-support.c:13:22

// The following invocations are in the 'named' group, and they follow
// the default invocation even if some of their ranges occur prior to the
// ranges from the default group because the groups are tested one-by-one:

// CHECK: invoking action 'local-rename':
// CHECK-NEXT: -selection={{.*}}tool-test-support.c:5:17

// CHECK: invoking action 'local-rename':
// CHECK-NEXT: -selection={{.*}}tool-test-support.c:11:20
