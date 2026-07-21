// Tests that we assign appropriate identifiers to indirect calls and targets
// for no-prototype functions which are treated as if they are variadic functions
// in generating their type strings for call graph section.

// RUN: %clang_cc1 -triple x86_64-unknown-linux -fexperimental-call-graph-section \
// RUN: -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefixes=WARN_NO_PROTOTYPE_ITANIUM %s

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fexperimental-call-graph-section \
// RUN: -emit-llvm -o /dev/null %s 2>&1 | FileCheck --check-prefixes=WARN_NO_PROTOTYPE_MS %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux -fexperimental-call-graph-section \
// RUN: -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,ITANIUM %s

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fexperimental-call-graph-section \
// RUN: -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,MS %s

// CHECK-LABEL: define {{(dso_local)?}} void @foo(
// CHECK-SAME: {{.*}} !callgraph [[F_TVOID:![0-9]+]]
void foo() {
}

// CHECK-LABEL: define {{(dso_local)?}} void @bar(
// CHECK-SAME: {{.*}} !callgraph [[F_TVOID]]
void bar() {
  void (*fp)() = foo;
  // ITANIUM: call {{.*}}, !callee_type [[F_TVOID_CT:![0-9]+]]
  // MS: call {{.*}}, !callee_type [[F_TVOID_CT:![0-9]+]]
  // WARN_NO_PROTOTYPE_ITANIUM: warning: indirect call to a function with no prototype 'void ()'; generating callee_type metadata as if calling a variadic function 'void (...)' (type string: _ZTSFvzE) [-Wcall-graph-section-no-prototype]
  // WARN_NO_PROTOTYPE_MS: warning: indirect call to a function with no prototype 'void ()'; generating callee_type metadata as if calling a variadic function 'void (...)' (type string: ?6AXZZ) [-Wcall-graph-section-no-prototype]
  fp();
}

struct my_struct;

// CHECK-LABEL: define {{(dso_local)?}} ptr @create_my_struct(
// CHECK-SAME: {{.*}} !callgraph [[F_TMY_STRUCT:![0-9]+]]
struct my_struct *create_my_struct() {
  return 0;
}

// CHECK-LABEL: define {{(dso_local)?}} void @test_struct_ptr_return(
// CHECK-SAME: {{.*}} !callgraph [[F_TVOID]]
void test_struct_ptr_return() {
  struct my_struct *(*fp)() = create_my_struct;
  // ITANIUM: call {{.*}}, !callee_type [[F_TMY_STRUCT_CT:![0-9]+]]
  // MS: call {{.*}}, !callee_type [[F_TMY_STRUCT_CT:![0-9]+]]
  // WARN_NO_PROTOTYPE_ITANIUM: warning: indirect call to a function with no prototype 'struct my_struct *()'; generating callee_type metadata as if calling a variadic function 'struct my_struct *(...)' (type string: _ZTSFP9my_structzE) [-Wcall-graph-section-no-prototype]
  // WARN_NO_PROTOTYPE_MS: warning: indirect call to a function with no prototype 'struct my_struct *()'; generating callee_type metadata as if calling a variadic function 'struct my_struct *(...)' (type string: ?6APEAUmy_struct@@ZZ) [-Wcall-graph-section-no-prototype]
  fp();
}

// CHECK-LABEL: define {{(dso_local)?}} i32 @baz(
// CHECK-SAME: {{.*}} !callgraph [[F_TINT:![0-9]+]]
int baz() {
  return 1;
}

// CHECK-LABEL: define {{(dso_local)?}} void @test_int_return(
// CHECK-SAME: {{.*}} !callgraph [[F_TVOID]]
void test_int_return() {
  int (*fp)() = baz;
  // ITANIUM: call {{.*}}, !callee_type [[F_TINT_CT:![0-9]+]]
  // MS: call {{.*}}, !callee_type [[F_TINT_CT:![0-9]+]]
  // WARN_NO_PROTOTYPE_ITANIUM: warning: indirect call to a function with no prototype 'int ()'; generating callee_type metadata as if calling a variadic function 'int (...)' (type string: _ZTSFizE) [-Wcall-graph-section-no-prototype]
  // WARN_NO_PROTOTYPE_MS: warning: indirect call to a function with no prototype 'int ()'; generating callee_type metadata as if calling a variadic function 'int (...)' (type string: ?6AHZZ) [-Wcall-graph-section-no-prototype]
  fp();
}

// CHECK-LABEL: define {{(dso_local)?}} void @test_no_proto_with_args(
// CHECK-SAME: {{.*}} !callgraph [[F_TVOID]]
void test_no_proto_with_args() {
  void (*fp)() = foo;
  // ITANIUM: call {{.*}}, !callee_type [[F_TVOID_CT:![0-9]+]]
  // MS: call {{.*}}, !callee_type [[F_TVOID_CT:![0-9]+]]
  // WARN_NO_PROTOTYPE_ITANIUM: warning: indirect call to a function with no prototype 'void ()'; generating callee_type metadata as if calling a variadic function 'void (...)' (type string: _ZTSFvzE) [-Wcall-graph-section-no-prototype]
  // WARN_NO_PROTOTYPE_MS: warning: indirect call to a function with no prototype 'void ()'; generating callee_type metadata as if calling a variadic function 'void (...)' (type string: ?6AXZZ) [-Wcall-graph-section-no-prototype]
  fp(1);
}

// ITANIUM: [[F_TVOID]] = !{!"_ZTSFvzE"}
// ITANIUM: [[F_TVOID_CT]] = !{[[F_TVOID:![0-9]+]]}
// ITANIUM: [[F_TMY_STRUCT]] = !{!"_ZTSFP9my_structzE"}
// ITANIUM: [[F_TMY_STRUCT_CT]] = !{[[F_TMY_STRUCT:![0-9]+]]}
// ITANIUM: [[F_TINT]] = !{!"_ZTSFizE"}
// ITANIUM: [[F_TINT_CT]] = !{[[F_TINT:![0-9]+]]}

// MS: [[F_TVOID]] = !{!"?6AXZZ"}
// MS: [[F_TVOID_CT]] = !{[[F_TVOID:![0-9]+]]}
// MS: [[F_TMY_STRUCT]] = !{!"?6APEAUmy_struct@@ZZ"}
// MS: [[F_TMY_STRUCT_CT]] = !{[[F_TMY_STRUCT:![0-9]+]]}
// MS: [[F_TINT]] = !{!"?6AHZZ"}
// MS: [[F_TINT_CT]] = !{[[F_TINT:![0-9]+]]}
