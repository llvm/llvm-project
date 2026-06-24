// RUN: %clang_cc1 -std=c++20 -emit-obj -debug-info-kind=standalone -dwarf-version=5 -triple x86_64-apple-darwin -o %t %s 
// RUN: llvm-dwarfdump %t | FileCheck %s

// CHECK: DW_TAG_structure_type
// CHECK-NEXT: DW_AT_calling_convention	(DW_CC_pass_by_value)
// CHECK-NEXT: DW_AT_name	("Foo<(lambda){}>")

template<auto T>
struct Foo {
};

Foo<[] {}> f;

auto func() { return f; }