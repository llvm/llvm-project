// RUN: clang-pseudo -grammar=cxx -source=%s --print-forest --forest-abbrev=false | FileCheck %s
class A {
    ;
// CHECK-NOT: member-declaration := ;
// CHECK: member-declaration := empty-declaration
// CHECK-NOT: member-declaration := ;
};
