// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -debug-info-kind=line-tables-only -S -emit-llvm -std=c++11 -o - %s | FileCheck --check-prefix LINUX %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -debug-info-kind=line-tables-only -gcodeview -S -emit-llvm -std=c++11 -o - %s | FileCheck --check-prefix MSVC %s

// Check that we emit type information for function scopes in line tables for
// CodeView.

namespace A {
void f() {}

struct S {
  static void m() {}
};
}

int main() {
  A::f();
  A::S::m();
  return 0;
  // MSVC:       !{{[0-9]+}} = distinct !DISubprogram(name: "f"
  // MSVC-SAME:     scope: [[SCOPE1:![0-9]+]]
  // MSVC-SAME:     )
  // MSVC:       [[SCOPE1]] = !DINamespace(name: "A", {{.*}})
  // MSVC:       !{{[0-9]+}} = distinct !DISubprogram(name: "m"
  // MSVC-SAME:     scope: [[SCOPE2:![0-9]+]]
  // MSVC-SAME:     )
  // MSVC:       [[SCOPE2]] = !DICompositeType(tag: DW_TAG_structure_type,
  // MSVC-SAME:     name: "S",
  // MSVC-SAME:     scope: [[SCOPE1]]
  // MSVC-SAME:     )

  // LINUX-NOT: !DINamespace
  // LINUX-NOT: !DICompositeType
  return 0;
}
