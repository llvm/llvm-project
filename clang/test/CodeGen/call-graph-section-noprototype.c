/// Tests that we assign appropriate identifiers to indirect calls for no-prototype
/// functions based on call site argument types.

// RUN: %clang_cc1 -triple x86_64-unknown-linux -fexperimental-call-graph-section \
// RUN: -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefixes=WARN_NO_PROTOTYPE_ITANIUM %s

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fexperimental-call-graph-section \
// RUN: -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefixes=WARN_NO_PROTOTYPE_MS %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux -fexperimental-call-graph-section \
// RUN: -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,ITANIUM %s

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fexperimental-call-graph-section \
// RUN: -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,MS %s

// CHECK-LABEL: define {{(dso_local)?}} void @foo(
// CHECK-SAME: {{.*}} !callgraph [[F_TVOID_NOPROTO:![0-9]+]]
void foo() {
}

// CHECK-LABEL: define {{(dso_local)?}} void @foo_with_proto(
// CHECK-SAME: {{.*}} !callgraph [[F_TVOID:![0-9]+]]
void foo_with_proto(void) {
}

// CHECK-LABEL: define {{(dso_local)?}} void @bar(
// CHECK-SAME: {{.*}} !callgraph [[F_TVOID_NOPROTO]]
void bar() {
  void (*fp)() = foo;
  // ITANIUM: call {{.*}}, !callee_type [[F_TVOID_CT:![0-9]+]]
  // MS: call {{.*}}, !callee_type [[F_TVOID_CT:![0-9]+]]
  // WARN_NO_PROTOTYPE_ITANIUM: warning: indirect call to a function with no prototype; generating type metadata for assumed prototype 'void (void)' (type string: _ZTSFvvE) [-Wcall-graph-section-no-prototype]
  // WARN_NO_PROTOTYPE_MS: warning: indirect call to a function with no prototype; generating type metadata for assumed prototype 'void (void)' (type string: ?6AXXZ) [-Wcall-graph-section-no-prototype]
  fp();
}

struct my_struct;

// CHECK-LABEL: define {{(dso_local)?}} ptr @create_my_struct(
// CHECK-SAME: {{.*}} !callgraph [[F_TMY_STRUCT_NOPROTO:![0-9]+]]
struct my_struct *create_my_struct() {
  return 0;
}

// CHECK-LABEL: define {{(dso_local)?}} void @test_struct_ptr_return(
// CHECK-SAME: {{.*}} !callgraph [[F_TVOID_NOPROTO]]
void test_struct_ptr_return() {
  struct my_struct *(*fp)() = create_my_struct;
  // ITANIUM: call {{.*}}, !callee_type [[F_TMY_STRUCT_CT:![0-9]+]]
  // MS: call {{.*}}, !callee_type [[F_TMY_STRUCT_CT:![0-9]+]]
  // WARN_NO_PROTOTYPE_ITANIUM: warning: indirect call to a function with no prototype; generating type metadata for assumed prototype 'struct my_struct *(void)' (type string: _ZTSFP9my_structvE) [-Wcall-graph-section-no-prototype]
  // WARN_NO_PROTOTYPE_MS: warning: indirect call to a function with no prototype; generating type metadata for assumed prototype 'struct my_struct *(void)' (type string: ?6APEAUmy_struct@@XZ) [-Wcall-graph-section-no-prototype]
  fp();
}

// CHECK-LABEL: define {{(dso_local)?}} i32 @baz(
// CHECK-SAME: {{.*}} !callgraph [[F_TINT_NOPROTO:![0-9]+]]
int baz() {
  return 1;
}

// CHECK-LABEL: define {{(dso_local)?}} void @test_int_return(
// CHECK-SAME: {{.*}} !callgraph [[F_TVOID_NOPROTO]]
void test_int_return() {
  int (*fp)() = baz;
  // ITANIUM: call {{.*}}, !callee_type [[F_TINT_CT:![0-9]+]]
  // MS: call {{.*}}, !callee_type [[F_TINT_CT:![0-9]+]]
  // WARN_NO_PROTOTYPE_ITANIUM: warning: indirect call to a function with no prototype; generating type metadata for assumed prototype 'int (void)' (type string: _ZTSFivE) [-Wcall-graph-section-no-prototype]
  // WARN_NO_PROTOTYPE_MS: warning: indirect call to a function with no prototype; generating type metadata for assumed prototype 'int (void)' (type string: ?6AHXZ) [-Wcall-graph-section-no-prototype]
  fp();
}

// CHECK-LABEL: define {{(dso_local)?}} void @test_no_proto_with_args(
// CHECK-SAME: {{.*}} !callgraph [[F_TVOID_NOPROTO]]
void test_no_proto_with_args() {
  void (*fp)() = foo;
  // ITANIUM: call {{.*}}, !callee_type [[F_TINT_ARG_CT:![0-9]+]]
  // MS: call {{.*}}, !callee_type [[F_TINT_ARG_CT:![0-9]+]]
  // WARN_NO_PROTOTYPE_ITANIUM: warning: indirect call to a function with no prototype; generating type metadata for assumed prototype 'void (int)' (type string: _ZTSFviE) [-Wcall-graph-section-no-prototype]
  // WARN_NO_PROTOTYPE_MS: warning: indirect call to a function with no prototype; generating type metadata for assumed prototype 'void (int)' (type string: ?6AXH@Z) [-Wcall-graph-section-no-prototype]
  fp(1);
}

/// Tests that multiple arguments passed to an unprototyped function pointer undergo
/// C default argument promotion (short -> int, float -> double) when reconstructing the prototype.
// CHECK-LABEL: define {{(dso_local)?}} void @test_promoted_args(
// CHECK-SAME: {{.*}} !callgraph [[F_TVOID_NOPROTO]]
void test_promoted_args() {
  void (*fp)() = foo;
  // ITANIUM: call {{.*}}, !callee_type [[F_TMULTI_ARG_CT:![0-9]+]]
  // MS: call {{.*}}, !callee_type [[F_TMULTI_ARG_CT:![0-9]+]]
  // WARN_NO_PROTOTYPE_ITANIUM: warning: indirect call to a function with no prototype; generating type metadata for assumed prototype 'void (int, double)' (type string: _ZTSFvidE) [-Wcall-graph-section-no-prototype]
  // WARN_NO_PROTOTYPE_MS: warning: indirect call to a function with no prototype; generating type metadata for assumed prototype 'void (int, double)' (type string: ?6AXHN@Z) [-Wcall-graph-section-no-prototype]
  fp((short)1, (float)2.0);
}

// ITANIUM: [[F_TVOID_NOPROTO]] = !{!"_ZTSFvE"}
// ITANIUM: [[F_TVOID]] = !{!"_ZTSFvvE"}
// ITANIUM: [[F_TVOID_CT]] = !{[[F_TVOID]]}
// ITANIUM: [[F_TMY_STRUCT_NOPROTO]] = !{!"_ZTSFP9my_structE"}
// ITANIUM: [[F_TMY_STRUCT_CT]] = !{[[F_TMY_STRUCT:![0-9]+]]}
// ITANIUM: [[F_TMY_STRUCT]] = !{!"_ZTSFP9my_structvE"}
// ITANIUM: [[F_TINT_NOPROTO]] = !{!"_ZTSFiE"}
// ITANIUM: [[F_TINT_CT]] = !{[[F_TINT:![0-9]+]]}
// ITANIUM: [[F_TINT]] = !{!"_ZTSFivE"}
// ITANIUM: [[F_TINT_ARG_CT]] = !{[[F_TINT_ARG:![0-9]+]]}
// ITANIUM: [[F_TINT_ARG]] = !{!"_ZTSFviE"}
// ITANIUM: [[F_TMULTI_ARG_CT]] = !{[[F_TMULTI_ARG:![0-9]+]]}
// ITANIUM: [[F_TMULTI_ARG]] = !{!"_ZTSFvidE"}

// MS: [[F_TVOID_NOPROTO]] = !{!"?6AX@Z"}
// MS: [[F_TVOID]] = !{!"?6AXXZ"}
// MS: [[F_TVOID_CT]] = !{[[F_TVOID]]}
// MS: [[F_TMY_STRUCT_NOPROTO]] = !{!"?6APEAUmy_struct@@@Z"}
// MS: [[F_TMY_STRUCT_CT]] = !{[[F_TMY_STRUCT:![0-9]+]]}
// MS: [[F_TMY_STRUCT]] = !{!"?6APEAUmy_struct@@XZ"}
// MS: [[F_TINT_NOPROTO]] = !{!"?6AH@Z"}
// MS: [[F_TINT_CT]] = !{[[F_TINT:![0-9]+]]}
// MS: [[F_TINT]] = !{!"?6AHXZ"}
// MS: [[F_TINT_ARG_CT]] = !{[[F_TINT_ARG:![0-9]+]]}
// MS: [[F_TINT_ARG]] = !{!"?6AXH@Z"}
// MS: [[F_TMULTI_ARG_CT]] = !{[[F_TMULTI_ARG:![0-9]+]]}
// MS: [[F_TMULTI_ARG]] = !{!"?6AXHN@Z"}
