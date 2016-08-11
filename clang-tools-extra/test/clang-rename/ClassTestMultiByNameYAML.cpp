class Foo1 { // CHECK: class Bar1
};

class Foo2 { // CHECK: class Bar2
};

// Test 1.
// RUN: clang-rename rename-all -input %S/Inputs/ClassTestMultiByNameYAMLRenameAll.yaml %s -- | sed 's,//.*,,' | FileCheck %s
// Test 2.
// RUN: clang-rename rename-all -input %S/Inputs/ClassTestMultiByNameYAMLRenameAt.yaml %s -- | sed 's,//.*,,' | FileCheck %s
