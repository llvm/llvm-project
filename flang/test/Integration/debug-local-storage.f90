! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s

! A named constant and a SAVE variable declared inside a procedure both have
! internal linkage. They are described in the scope of that procedure, are
! local to the compile unit and carry no linkage name. The `name` field being
! followed directly by `scope` is what checks that no linkage name is emitted.

! CHECK-DAG: ![[I4:.*]] = !DIBasicType(name: "integer(kind=4)", size: 32, encoding: DW_ATE_signed)
! CHECK-DAG: ![[SUB:.*]] = distinct !DISubprogram(name: "counter_fn"{{.*}})

integer function counter_fn()
! CHECK-DAG: ![[Q:.*]] = distinct !DIGlobalVariable(name: "q", scope: ![[SUB]], file: !{{[0-9]+}}, line: [[@LINE+2]], type: ![[I4]], isLocal: true, isDefinition: true)
! CHECK-DAG: !DIGlobalVariableExpression(var: ![[Q]], expr: !DIExpression())
  integer, parameter :: q = 7
! CHECK-DAG: ![[COUNT:.*]] = distinct !DIGlobalVariable(name: "counter", scope: ![[SUB]], file: !{{[0-9]+}}, line: [[@LINE+2]], type: ![[I4]], isLocal: true, isDefinition: true)
! CHECK-DAG: !DIGlobalVariableExpression(var: ![[COUNT]], expr: !DIExpression())
  integer, save :: counter = 0
  counter = counter + 1
  counter_fn = q + counter
end function counter_fn
