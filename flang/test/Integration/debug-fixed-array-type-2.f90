! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s

program mn

  integer d1(3)
  integer d2(2, 5)
  real d3(6, 8, 7)

  i8 = fn1(d1, d2, d3)
contains
  function fn1(a1, b1, c1) result (res)
    integer a1(3)
    integer b1(2, 5)
    real c1(6, 8, 7)
    integer res
    res = a1(1) + b1(1,2) + c1(3, 3, 4)
  end function

end program

! CHECK-DAG: ![[INT:.*]] = !DIBasicType(name: "integer", size: 32, encoding: DW_ATE_signed)
! CHECK-DAG: ![[REAL:.*]] = !DIBasicType(name: "real", size: 32, encoding: DW_ATE_float)
! CHECK-DAG: ![[R1:.*]] = !DISubrange(count: 3, lowerBound: 1)
! CHECK-DAG: ![[SUB1:.*]] = !{![[R1]]}
! CHECK-DAG: ![[D1TY:.*]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[INT]], elements: ![[SUB1]])
! CHECK-DAG: !DILocalVariable(name: "d1"{{.*}}type: ![[D1TY]])

! CHECK-DAG: ![[R21:.*]] = !DISubrange(count: 2, lowerBound: 1)
! CHECK-DAG: ![[R22:.*]] = !DISubrange(count: 5, lowerBound: 1)
! CHECK-DAG: ![[SUB2:.*]] = !{![[R21]], ![[R22]]}
! CHECK-DAG: ![[D2TY:.*]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[INT]], elements: ![[SUB2]])
! CHECK-DAG: !DILocalVariable(name: "d2"{{.*}}type: ![[D2TY]])

! CHECK-DAG: ![[R31:.*]] = !DISubrange(count: 6, lowerBound: 1)
! CHECK-DAG: ![[R32:.*]] = !DISubrange(count: 8, lowerBound: 1)
! CHECK-DAG: ![[R33:.*]] = !DISubrange(count: 7, lowerBound: 1)
! CHECK-DAG: ![[SUB3:.*]] = !{![[R31]], ![[R32]], ![[R33]]}
! CHECK-DAG: ![[D3TY:.*]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[REAL]], elements: ![[SUB3]])
! CHECK-DAG: !DILocalVariable(name: "d3"{{.*}}type: ![[D3TY]])

! CHECK-DAG: !DILocalVariable(name: "a1", arg: 1{{.*}}type: ![[D1TY]])
! CHECK-DAG: !DILocalVariable(name: "b1", arg: 2{{.*}}type: ![[D2TY]])
! CHECK-DAG: !DILocalVariable(name: "c1", arg: 3{{.*}}type: ![[D3TY]])
