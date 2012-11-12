// RUN: %clang --analyze %s -o /dev/null -Xclang -analyzer-checker=debug.ConfigDumper > %t 2>&1
// RUN: FileCheck --input-file=%t %s

void bar() {}
void foo() { bar(); }

class Foo {
public:
	void bar() {}
	void foo() { bar(); }
};

// CHECK: [config]
// CHECK-NEXT: c++-inlining = methods
// CHECK-NEXT: c++-stdlib-inlining = true
// CHECK-NEXT: c++-template-inlining = true
// CHECK-NEXT: cfg-temporary-dtors = false
// CHECK-NEXT: faux-bodies = true
// CHECK-NEXT: graph-trim-interval = 1000
// CHECK-NEXT: ipa-always-inline-size = 3
// CHECK-NEXT: [stats]
// CHECK-NEXT: num-entries = 7
