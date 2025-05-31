! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s

module test
  integer d1(3)
  integer d2(1:4, -1:3)
  real d3(-2:6, 0:5, 3:7)
end

program mn
  use test

  i8 = fn1(d1, d2, d3)
contains
  function fn1(a1, b1, c1) result (res)
    integer a1(3)
    integer b1(-1:0, 5:9)
    real c1(-2:6, 0:5, 3:7)
    integer res
    res = a1(1) + b1(0,6) + c1(3, 3, 4)
  end function

end program

! CHECK-DAG: ![[INT:.*]] = !DIBasicType(name: "integer", size: 32, encoding: DW_ATE_signed)
! CHECK-DAG: ![[REAL:.*]] = !DIBasicType(name: "real", size: 32, encoding: DW_ATE_float)
! CHECK-DAG: ![[R1:.*]] = !DISubrange(count: 3)
! CHECK-DAG: ![[SUB1:.*]] = !{![[R1]]}
! CHECK-DAG: ![[D1TY:.*]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[INT]], elements: ![[SUB1]])

! CHECK-DAG: ![[R21:.*]] = !DISubrange(count: 4)
! CHECK-DAG: ![[R22:.*]] = !DISubrange(count: 5, lowerBound: -1)
! CHECK-DAG: ![[SUB2:.*]] = !{![[R21]], ![[R22]]}
! CHECK-DAG: ![[D2TY:.*]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[INT]], elements: ![[SUB2]])

! CHECK-DAG: ![[R31:.*]] = !DISubrange(count: 9, lowerBound: -2)
! CHECK-DAG: ![[R32:.*]] = !DISubrange(count: 6, lowerBound: 0)
! CHECK-DAG: ![[R33:.*]] = !DISubrange(count: 5, lowerBound: 3)
! CHECK-DAG: ![[SUB3:.*]] = !{![[R31]], ![[R32]], ![[R33]]}
! CHECK-DAG: ![[D3TY:.*]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[REAL]], elements: ![[SUB3]])

! CHECK-DAG: ![[B11:.*]] = !DISubrange(count: 2, lowerBound: -1)
! CHECK-DAG: ![[B12:.*]] = !DISubrange(count: 5, lowerBound: 5)
! CHECK-DAG: ![[B1:.*]] = !{![[B11]], ![[B12]]}
! CHECK-DAG: ![[B1TY:.*]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[INT]], elements: ![[B1]])

! CHECK-DAG: {{.*}}!DIGlobalVariable(name: "d1"{{.*}}type: ![[D1TY]]{{.*}})
! CHECK-DAG: {{.*}}!DIGlobalVariable(name: "d2"{{.*}}type: ![[D2TY]]{{.*}})
! CHECK-DAG: {{.*}}!DIGlobalVariable(name: "d3"{{.*}}type: ![[D3TY]]{{.*}})

! CHECK-DAG: !DILocalVariable(name: "a1", arg: 1{{.*}}type: ![[D1TY]])
! CHECK-DAG: !DILocalVariable(name: "b1", arg: 2{{.*}}type: ![[B1TY]])
! CHECK-DAG: !DILocalVariable(name: "c1", arg: 3{{.*}}type: ![[D3TY]])
