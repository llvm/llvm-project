/// Tests that function definitions without a prototype (C89 empty parameter list or K&R declarations)
/// reconstruct parameter types with default argument promotions and produce type identifiers that
/// match both:
/// - Their prototyped equivalents on the definition side.
/// - Indirect callsites using unprototyped function pointers.

// RUN: %clang_cc1 -triple x86_64-unknown-linux -fexperimental-call-graph-section \
// RUN: -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,ITANIUM %s

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fexperimental-call-graph-section \
// RUN: -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,MS %s

/// Forward declaration without prototype (pure declaration in this TU).
/// Because there is no definition or prototype in this TU, no !callgraph metadata is attached.
// CHECK-LABEL: declare {{.*}}void @decl_only(...)
void decl_only();

void use_decl(void) {
  void (*fp)() = decl_only;
}

/// Void parameter list: C89 parameterless definition and C prototyped (void) definition
/// must produce the same type identifier.
// CHECK-LABEL: define {{(dso_local)?}} void @proto_void(
// CHECK-SAME: {{.*}} !callgraph [[F_TVOID:![0-9]+]]
void proto_void(void) {}

// CHECK-LABEL: define {{(dso_local)?}} void @c89_void(
// CHECK-SAME: {{.*}} !callgraph [[F_TVOID]]
void c89_void() {}

/// Single argument promotion: K&R int definition and K&R short definition (promoted to int)
/// must produce the same type identifier.
// CHECK-LABEL: define {{(dso_local)?}} void @knr_int(
// CHECK-SAME: {{.*}} !callgraph [[F_TINT_ONE:![0-9]+]]
void knr_int(i)
  int i;
{}

// CHECK-LABEL: define {{(dso_local)?}} void @knr_promoted_int(
// CHECK-SAME: {{.*}} !callgraph [[F_TINT_ONE]]
void knr_promoted_int(i)
  short i;
{}

/// Multi-argument promotions: prototyped (int, double), K&R (int, double), and K&R (short, float)
/// (promoted to int, double) must all produce the same type identifier.
// CHECK-LABEL: define {{(dso_local)?}} void @proto_int_double(
// CHECK-SAME: {{.*}} !callgraph [[F_TINT_DOUBLE:![0-9]+]]
void proto_int_double(int a, double b) {}

// CHECK-LABEL: define {{(dso_local)?}} void @knr_int_double(
// CHECK-SAME: {{.*}} !callgraph [[F_TINT_DOUBLE]]
void knr_int_double(a, b)
  int a;
  double b;
{}

// CHECK-LABEL: define {{(dso_local)?}} void @knr_promoted_multi(
// CHECK-SAME: {{.*}} !callgraph [[F_TINT_DOUBLE]]
void knr_promoted_multi(a, b)
  short a;
  float b;
{}

/// Struct pointer return: C89 parameterless and prototyped (void) return types must match.
struct my_struct;

// CHECK-LABEL: define {{(dso_local)?}} ptr @proto_my_struct(
// CHECK-SAME: {{.*}} !callgraph [[F_TMY_STRUCT:![0-9]+]]
struct my_struct *proto_my_struct(void) { return 0; }

// CHECK-LABEL: define {{(dso_local)?}} ptr @c89_my_struct(
// CHECK-SAME: {{.*}} !callgraph [[F_TMY_STRUCT]]
struct my_struct *c89_my_struct() { return 0; }

/// Callsite-to-definition equivalence: indirect calls using unprototyped function pointers
/// must generate !callee_type metadata referencing the normalized definition type identifiers.
void test_indirect_calls(void) {
  // CHECK: call void {{.*}}, !callee_type [[F_TVOID_CT:![0-9]+]]
  void (*fp_void)() = c89_void;
  fp_void();

  // CHECK: call void {{.*}}, !callee_type [[F_TINT_DOUBLE_CT:![0-9]+]]
  void (*fp_multi)() = knr_promoted_multi;
  fp_multi((short)1, (float)2.0);
}

// ITANIUM: [[F_TVOID]] = !{!"_ZTSFvvE"}
// ITANIUM: [[F_TINT_ONE]] = !{!"_ZTSFviE"}
// ITANIUM: [[F_TINT_DOUBLE]] = !{!"_ZTSFvidE"}
// ITANIUM: [[F_TMY_STRUCT]] = !{!"_ZTSFP9my_structvE"}

// MS: [[F_TVOID]] = !{!"?6AXXZ"}
// MS: [[F_TINT_ONE]] = !{!"?6AXH@Z"}
// MS: [[F_TINT_DOUBLE]] = !{!"?6AXHN@Z"}
// MS: [[F_TMY_STRUCT]] = !{!"?6APEAUmy_struct@@XZ"}

// CHECK: [[F_TVOID_CT]] = !{[[F_TVOID]]}
// CHECK: [[F_TINT_DOUBLE_CT]] = !{[[F_TINT_DOUBLE]]}
