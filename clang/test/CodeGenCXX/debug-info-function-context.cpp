// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -triple x86_64-pc-linux-gnu %s -o - | FileCheck %s

struct C {
  void member_function();
  static int static_member_function();
  static int static_member_variable;
};

int C::static_member_variable = 0;

void C::member_function() { static_member_variable = 0; }

int C::static_member_function() { return static_member_variable; }

C global_variable;

int global_function() { return -1; }

namespace ns {
void global_namespace_function() { global_variable.member_function(); }
int global_namespace_variable = 1;
}

// Check that the functions that belong to C have C as a context and the
// functions that belong to the namespace have it as a context, and the global
// function has the file as a context.

// CHECK: ![[FILE:[0-9]+]] = !DIFile(filename: "{{.*}}context.cpp",
// CHECK: ![[C:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C",
// CHECK: ![[NS:.*]] = !DINamespace(name: "ns"
// CHECK: !DISubprogram(name: "member_function",{{.*}} scope: ![[C]],{{.*}} isDefinition: true

// CHECK: !DISubprogram(name: "static_member_function",{{.*}} scope: ![[C]],{{.*}} isDefinition: true

// CHECK: !DISubprogram(name: "global_function",{{.*}} scope: ![[FILE]],{{.*}} isDefinition: true

// CHECK: !DISubprogram(name: "global_namespace_function",{{.*}} scope: ![[NS]],{{.*}} isDefinition: true
