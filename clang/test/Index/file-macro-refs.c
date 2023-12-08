#define FOO

FOO
FOO

// RUN: c-index-test -file-refs-at=%s:3:2 %s | FileCheck %s
// RUN: env CINDEXTEST_EDITING=1 c-index-test -file-refs-at=%s:3:2 %s | FileCheck %s

// CHECK:      macro expansion=FOO:1:9
// CHECK-NEXT: macro definition=FOO =[1:9 - 1:12]
// CHECK-NEXT: macro expansion=FOO:1:9 =[3:1 - 3:4]
// CHECK-NEXT: macro expansion=FOO:1:9 =[4:1 - 4:4]
