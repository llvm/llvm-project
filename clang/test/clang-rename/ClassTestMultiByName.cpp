class Foo1 { // CHECK: class Bar1
};

class Foo2 { // CHECK: class Bar2
};

// Test 1.
// RUN: clang-rename -qualified-name=Foo1 -new-name=Bar1 -qualified-name=Foo2 -new-name=Bar2 %s -- | sed 's,//.*,,' | FileCheck %s
