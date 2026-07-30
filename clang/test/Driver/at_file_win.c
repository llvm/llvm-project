// RUN: %clang --rsp-quoting=windows -E %s @%s.args -o %t.log
// RUN: FileCheck --input-file=%t.log %s

// CHECK: bar1
// CHECK-NEXT: bar2 zed2
// CHECK-NEXT: bar3 zed3
// CHECK-NEXT: bar4 zed4
// CHECK-NEXT: bar5 zed5
// CHECK-NEXT: 'bar6 zed6'
// CHECK-NEXT: 'bar7 zed7'
// CHECK-NEXT: foo8bar8zed8
// CHECK-NEXT: foo9\'bar9\'zed9
// CHECK-NEXT: foo10"bar10"zed10
// CHECK: bar
// CHECK: zed12
// CHECK: one\two
// CHECK: c:\foo\bar.c

foo1
foo2
foo3
foo4
foo5
foo6
foo7
foo8
foo9
foo10
#ifdef foo11
bar
#endif
foo12
foo13
foo14
