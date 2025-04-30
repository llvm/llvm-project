!RUN: %flang -g %s -S -emit-llvm -o - | FileCheck %s

!CHECK-DAG: DIModule(scope: !{{.*}}, name: "dummy", file: !{{.*}}, line: 9)
!CHECK-DAG: ![[FOO_NODE:.*]] = {{.*}} !DIGlobalVariable(name: "foo", {{.*}}
!CHECK-DAG: ![[BAR_NODE:.*]] = {{.*}} !DIGlobalVariable(name: "bar", {{.*}}
!CHECK-DAG: DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !{{.*}}, entity: ![[FOO_NODE]], file: !{{.*}}, line: 14)
!CHECK-NOT: DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !{{.*}}, entity: ![[BAR_NODE]], file: !{{.*}}, line: 14)

MODULE dummy !line no. 9
      INTEGER :: foo
      INTEGER :: bar
END MODULE dummy

PROGRAM main
USE dummy, ONLY: foo

END PROGRAM
