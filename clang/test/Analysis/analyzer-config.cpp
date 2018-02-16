// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 %s -o /dev/null -analyzer-checker=core,osx.cocoa,debug.ConfigDumper -analyzer-max-loop 34 > %t 2>&1
// RUN: FileCheck --input-file=%t %s

void bar() {}
void foo() {
  // Call bar 33 times so max-times-inline-large is met and
  // min-blocks-for-inline-large is checked
  for (int i = 0; i < 34; ++i) {
    bar();
  }
}

class Foo {
public:
  ~Foo() {}
  void baz();
	void bar() { const Foo &f = Foo(); }
	void foo() { bar(); }
};

// CHECK: [config]
// CHECK-NEXT: c++-container-inlining = false
// CHECK-NEXT: c++-inlining = destructors
// CHECK-NEXT: c++-shared_ptr-inlining = false
// CHECK-NEXT: c++-stdlib-inlining = true
// CHECK-NEXT: c++-template-inlining = true
// CHECK-NEXT: cfg-conditional-static-initializers = true
// CHECK-NEXT: cfg-implicit-dtors = true
// CHECK-NEXT: cfg-lifetime = false
// CHECK-NEXT: cfg-loopexit = false
// CHECK-NEXT: cfg-rich-constructors = true
// CHECK-NEXT: cfg-temporary-dtors = false
// CHECK-NEXT: exploration_strategy = dfs
// CHECK-NEXT: faux-bodies = true
// CHECK-NEXT: graph-trim-interval = 1000
// CHECK-NEXT: inline-lambdas = true
// CHECK-NEXT: ipa = dynamic-bifurcate
// CHECK-NEXT: ipa-always-inline-size = 3
// CHECK-NEXT: leak-diagnostics-reference-allocation = false
// CHECK-NEXT: max-inlinable-size = 100
// CHECK-NEXT: max-nodes = 225000
// CHECK-NEXT: max-times-inline-large = 32
// CHECK-NEXT: min-cfg-size-treat-functions-as-large = 14
// CHECK-NEXT: mode = deep
// CHECK-NEXT: region-store-small-struct-limit = 2
// CHECK-NEXT: serialize-stats = false
// CHECK-NEXT: unroll-loops = false
// CHECK-NEXT: widen-loops = false
// CHECK-NEXT: [stats]
// CHECK-NEXT: num-entries = 27
