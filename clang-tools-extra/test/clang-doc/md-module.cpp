// THIS IS A GENERATED TEST. DO NOT EDIT.
// To regenerate, see clang-doc/gen_test.py docstring.
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"

export module M;

int moduleFunction(int x); // ModuleLinkage

static int staticModuleFunction(int x); // ModuleInternalLinkage

export double exportedModuleFunction(double y, int z); // ExternalLinkage

// RUN: clang-doc --format=md --doxygen --public --extra-arg=-fmodules-ts -p %t %t/test.cpp -output=%t/docs


// RUN: cat %t/docs/./GlobalNamespace.md | FileCheck %s --check-prefix CHECK-0
// CHECK-0: # Global Namespace
// CHECK-0: ## Functions
// CHECK-0: ### moduleFunction
// CHECK-0: *int moduleFunction(int x)*
// CHECK-0: ### exportedModuleFunction
// CHECK-0: *double exportedModuleFunction(double y, int z)*
