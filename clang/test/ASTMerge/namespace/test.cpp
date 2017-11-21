// RUN: %clang_cc1 -emit-pch -std=c++1z -o %t.1.ast %S/Inputs/namespace1.cpp
// RUN: %clang_cc1 -emit-pch -std=c++1z -o %t.2.ast %S/Inputs/namespace2.cpp
// RUN: not %clang_cc1 -std=c++1z -ast-merge %t.1.ast -ast-merge %t.2.ast -fsyntax-only %s 2>&1 | FileCheck %s

static_assert(TestAliasName::z == 4);
static_assert(ContainsInline::z == 10);

void testImport() {
  typedef TestUnresolvedTypenameAndValueDecls::Derived<int> Imported;
  Imported a; // Successfull instantiation
  static_assert(sizeof(Imported::foo) == sizeof(int));
  static_assert(sizeof(TestUnresolvedTypenameAndValueDecls::Derived<double>::NewUnresolvedUsingType) == sizeof(double));
}


// CHECK: namespace2.cpp:16:17: error: external variable 'z' declared with incompatible types in different translation units ('double' vs. 'float')
// CHECK: namespace1.cpp:16:16: note: declared here with type 'float'
