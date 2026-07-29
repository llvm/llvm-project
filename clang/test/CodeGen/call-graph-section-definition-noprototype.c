// RUN: %clang_cc1 -triple x86_64-unknown-linux -fexperimental-call-graph-section \
// RUN: -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,ITANIUM %s

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fexperimental-call-graph-section \
// RUN: -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,MS %s

// Tests that function definitions specified without a prototype (C89 empty parameter list or K&R declarations)
// generate !callgraph metadata based on reconstructed parameter types (with default argument promotions).

// Forward declaration without prototype (pure declaration in this TU).
// Because there is no definition or prototype in this TU, no !callgraph metadata is attached to @decl_only declaration.
// CHECK-LABEL: declare {{.*}}void @decl_only(...)
void decl_only();

void use_decl() {
  void (*fp)() = decl_only;
}

// C89 definition with no parameters: reconstructed prototype void (void).
// CHECK-LABEL: define {{(dso_local)?}} void @foo(
// CHECK-SAME: {{.*}} !callgraph [[F_TVOID:![0-9]+]]
void foo() {
}

// Function returning struct pointer with no parameters: reconstructed prototype struct my_struct *(void).
struct my_struct;
// CHECK-LABEL: define {{(dso_local)?}} ptr @create_my_struct(
// CHECK-SAME: {{.*}} !callgraph [[F_TMY_STRUCT:![0-9]+]]
struct my_struct *create_my_struct() {
  return 0;
}

// Function returning int with no parameters: reconstructed prototype int (void).
// CHECK-LABEL: define {{(dso_local)?}} i32 @baz(
// CHECK-SAME: {{.*}} !callgraph [[F_TINT:![0-9]+]]
int baz() {
  return 1;
}

// K&R function definition: reconstructed prototype void (int, int) with promoted short -> int.
// CHECK-LABEL: define {{(dso_local)?}} void @knr_func(
// CHECK-SAME: {{.*}} !callgraph [[F_TKNR:![0-9]+]]
void knr_func(a, b)
  int a;
  short b;
{
}

// ITANIUM: [[F_TVOID]] = !{!"_ZTSFvvE"}
// ITANIUM: [[F_TMY_STRUCT]] = !{!"_ZTSFP9my_structvE"}
// ITANIUM: [[F_TINT]] = !{!"_ZTSFivE"}
// ITANIUM: [[F_TKNR]] = !{!"_ZTSFviiE"}

// MS: [[F_TVOID]] = !{!"?6AXXZ"}
// MS: [[F_TMY_STRUCT]] = !{!"?6APEAUmy_struct@@XZ"}
// MS: [[F_TINT]] = !{!"?6AHXZ"}
// MS: [[F_TKNR]] = !{!"?6AXHH@Z"}
