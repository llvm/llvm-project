! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck  %s

program mn
  complex(kind=4) :: c4
  complex(kind=8) :: c8
  complex(kind=16) :: r
  r = fn1(c4, c8)
  print *, r
contains
  function fn1(a, b) result (c)
    complex(kind=4), intent(in) :: a
    complex(kind=8), intent(in) :: b
    complex(kind=16) :: c
    c = a + b
  end function
end program

! CHECK-DAG: ![[C4:.*]] = !DIBasicType(name: "complex", size: 64, encoding: DW_ATE_complex_float)
! CHECK-DAG: ![[C8:.*]] = !DIBasicType(name: "complex", size: 128, encoding: DW_ATE_complex_float)
! CHECK-DAG: ![[C16:.*]] = !DIBasicType(name: "complex", size: 256, encoding: DW_ATE_complex_float)
! CHECK-DAG: !DILocalVariable(name: "c4"{{.*}}type: ![[C4]])
! CHECK-DAG: !DILocalVariable(name: "c8"{{.*}}type: ![[C8]])
! CHECK-DAG: !DILocalVariable(name: "r"{{.*}}type: ![[C16]])
! CHECK-DAG: !DILocalVariable(name: "a"{{.*}}type: ![[C4]])
! CHECK-DAG: !DILocalVariable(name: "b"{{.*}}type: ![[C8]])
! CHECK-DAG: !DILocalVariable(name: "c"{{.*}}type: ![[C16]])
