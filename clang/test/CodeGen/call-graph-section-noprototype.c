/// Tests that we assign appropriate identifiers to indirect calls for no-prototype
/// functions based on call site argument types.

// RUN: %clang_cc1 -triple x86_64-unknown-linux -fexperimental-call-graph-section \
// RUN: -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefix=WARN %s

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fexperimental-call-graph-section \
// RUN: -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefix=WARN %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux -fexperimental-call-graph-section \
// RUN: -emit-llvm -o - %s | FileCheck --check-prefix=CHECK %s

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fexperimental-call-graph-section \
// RUN: -emit-llvm -o - %s | FileCheck --check-prefix=CHECK %s

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
  // CHECK: call {{.*}}, !callee_type [[F_TVOID_CT:![0-9]+]]
  // WARN: warning: indirect call to a function with no prototype; generating type metadata for assumed prototype 'void (void)' (type string: {{.*}}) [-Wcall-graph-section-no-prototype]
  fp();
}

struct my_struct;

// CHECK-LABEL: define {{(dso_local)?}} ptr @create_my_struct(
// CHECK-SAME: {{.*}} !callgraph [[F_TMY_STRUCT_NOPROTO:![0-9]+]]
struct my_struct *create_my_struct() {
  return 0;
}

// CHECK-LABEL: define {{(dso_local)?}} ptr @create_my_struct_with_proto(
// CHECK-SAME: {{.*}} !callgraph [[F_TMY_STRUCT:![0-9]+]]
struct my_struct *create_my_struct_with_proto(void) {
  return 0;
}

// CHECK-LABEL: define {{(dso_local)?}} void @test_struct_ptr_return(
// CHECK-SAME: {{.*}} !callgraph [[F_TVOID_NOPROTO]]
void test_struct_ptr_return() {
  struct my_struct *(*fp)() = create_my_struct;
  // CHECK: call {{.*}}, !callee_type [[F_TMY_STRUCT_CT:![0-9]+]]
  // WARN: warning: indirect call to a function with no prototype; generating type metadata for assumed prototype 'struct my_struct *(void)' (type string: {{.*}}) [-Wcall-graph-section-no-prototype]
  fp();
}

// CHECK-LABEL: define {{(dso_local)?}} i32 @baz(
// CHECK-SAME: {{.*}} !callgraph [[F_TINT_NOPROTO:![0-9]+]]
int baz() {
  return 1;
}

// CHECK-LABEL: define {{(dso_local)?}} i32 @baz_with_proto(
// CHECK-SAME: {{.*}} !callgraph [[F_TINT:![0-9]+]]
int baz_with_proto(void) {
  return 1;
}

// CHECK-LABEL: define {{(dso_local)?}} void @test_int_return(
// CHECK-SAME: {{.*}} !callgraph [[F_TVOID_NOPROTO]]
void test_int_return() {
  int (*fp)() = baz;
  // CHECK: call {{.*}}, !callee_type [[F_TINT_CT:![0-9]+]]
  // WARN: warning: indirect call to a function with no prototype; generating type metadata for assumed prototype 'int (void)' (type string: {{.*}}) [-Wcall-graph-section-no-prototype]
  fp();
}

// CHECK-LABEL: define {{(dso_local)?}} void @foo_with_int_proto(
// CHECK-SAME: {{.*}} !callgraph [[F_TINT_ARG:![0-9]+]]
void foo_with_int_proto(int a) {
}

// CHECK-LABEL: define {{(dso_local)?}} void @test_no_proto_with_args(
// CHECK-SAME: {{.*}} !callgraph [[F_TVOID_NOPROTO]]
void test_no_proto_with_args() {
  void (*fp)() = foo;
  // CHECK: call {{.*}}, !callee_type [[F_TINT_ARG_CT:![0-9]+]]
  // WARN: warning: indirect call to a function with no prototype; generating type metadata for assumed prototype 'void (int)' (type string: {{.*}}) [-Wcall-graph-section-no-prototype]
  fp(1);
}

// CHECK-LABEL: define {{(dso_local)?}} void @foo_with_promoted_proto(
// CHECK-SAME: {{.*}} !callgraph [[F_TMULTI_ARG:![0-9]+]]
void foo_with_promoted_proto(int a, double b) {
}

/// Tests that multiple arguments passed to an unprototyped function pointer undergo
/// C default argument promotion (short -> int, float -> double) when reconstructing the prototype.
// CHECK-LABEL: define {{(dso_local)?}} void @test_promoted_args(
// CHECK-SAME: {{.*}} !callgraph [[F_TVOID_NOPROTO]]
void test_promoted_args() {
  void (*fp)() = foo;
  // CHECK: call {{.*}}, !callee_type [[F_TMULTI_ARG_CT:![0-9]+]]
  // WARN: warning: indirect call to a function with no prototype; generating type metadata for assumed prototype 'void (int, double)' (type string: {{.*}}) [-Wcall-graph-section-no-prototype]
  fp((short)1, (float)2.0);
}

// CHECK: [[F_TVOID_CT]] = !{[[F_TVOID]]}
// CHECK: [[F_TMY_STRUCT_CT]] = !{[[F_TMY_STRUCT]]}
// CHECK: [[F_TINT_CT]] = !{[[F_TINT]]}
// CHECK: [[F_TINT_ARG_CT]] = !{[[F_TINT_ARG]]}
// CHECK: [[F_TMULTI_ARG_CT]] = !{[[F_TMULTI_ARG]]}
