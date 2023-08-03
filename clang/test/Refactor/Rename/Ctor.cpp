class Foo {                  // CHECK: rename [[@LINE]]:7 -> [[@LINE]]:10
public:
  Foo();                     // CHECK: rename [[@LINE]]:3 -> [[@LINE]]:6
  Foo(int x, int y);         // CHECK: rename [[@LINE]]:3 -> [[@LINE]]:6
  ~Foo();                    // CHECK: rename [[@LINE]]:4 -> [[@LINE]]:7
};

Foo::Foo() {}               // CHECK: rename [[@LINE]]:1 -> [[@LINE]]:4
// CHECK: rename [[@LINE-1]]:6 -> [[@LINE-1]]:9

Foo::Foo(int x, int y) { }  // CHECK: rename [[@LINE]]:1 -> [[@LINE]]:4
// CHECK: rename [[@LINE-1]]:6 -> [[@LINE-1]]:9

Foo::~Foo() {}              // CHECK: rename [[@LINE]]:1 -> [[@LINE]]:4
// CHECK: rename [[@LINE-1]]:7 -> [[@LINE-1]]:10

Foo f(const Foo &Other) {   // CHECK: rename [[@LINE]]:1 -> [[@LINE]]:4
                            // CHECK: rename [[@LINE-1]]:13 -> [[@LINE-1]]:16
  return Foo(Other);        // CHECK: rename [[@LINE]]:10 -> [[@LINE]]:13
}

// Declarations.
// RUN: clang-refactor-test rename-initiate -at=%s:3:3 -at=%s:4:3 -at=%s:5:4 -new-name=Bar %s | FileCheck %s

// Definitions.
// RUN: clang-refactor-test rename-initiate -at=%s:8:6 -at=%s:11:6 -at=%s:14:7 -new-name=Bar %s | FileCheck %s

// Implicit copy constructor.
// RUN: clang-refactor-test rename-initiate -at=%s:19:10 -new-name=Bar %s | FileCheck %s
