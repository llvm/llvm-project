// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -triple x86_64-pc-linux-gnu %s -fdebug-compilation-dir=%S \
// RUN:     -dwarf-version=5 -main-file-name function-context.cpp  -o - | FileCheck %s

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

// Generate the artificial global functions to initialize a global.
int global_initialized_variable = C::static_member_function();

// Check that the functions that belong to C have C as a context and the
// functions that belong to the namespace have it as a context, and the global
// functions (user-defined and artificial) have the file as a context.

// The first DIFile is for the CU, the second is what everything else uses.
// We're using DWARF v5 so both should have MD5 checksums.
// CHECK: !DIFile(filename: "{{.*}}context.cpp",{{.*}} checksumkind: CSK_MD5, checksum: [[CKSUM:".*"]]
// CHECK: ![[FILE:[0-9]+]] = !DIFile(filename: "{{.*}}context.cpp",{{.*}} checksumkind: CSK_MD5, checksum: [[CKSUM]]
// CHECK: ![[C:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C",
// CHECK: ![[NS:.*]] = !DINamespace(name: "ns"
// CHECK: !DISubprogram(name: "member_function",{{.*}} scope: ![[C]],{{.*}} DISPFlagDefinition

// CHECK: !DISubprogram(name: "static_member_function",{{.*}} scope: ![[C]],{{.*}} DISPFlagDefinition

// CHECK: !DISubprogram(name: "global_function",{{.*}} scope: ![[FILE]],{{.*}} DISPFlagDefinition

// CHECK: !DISubprogram(name: "global_namespace_function",{{.*}} scope: ![[NS]],{{.*}} DISPFlagDefinition

// CHECK: !DISubprogram(name: "__cxx_global_var_init",{{.*}} scope: ![[FILE]],{{.*}} DISPFlagDefinition
// CHECK: !DISubprogram(linkageName: "_GLOBAL__sub_I_{{.*}}",{{.*}} scope: ![[FILE]],{{.*}} DISPFlagDefinition
