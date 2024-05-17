! RUN: %flang_fc1 -emit-fir -debug-info-kind=standalone -mmlir --mlir-print-debuginfo %s -o - | \
! RUN: fir-opt --cg-rewrite="preserve-declare=true" --mlir-print-debuginfo | fir-opt --add-debug-info --mlir-print-debuginfo | FileCheck %s
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck  --check-prefix=LLVM %s

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

! CHECK-DAG: #[[CMPX4:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "complex", sizeInBits = 64, encoding = DW_ATE_complex_float>
! CHECK-DAG: #[[CMPX8:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "complex", sizeInBits = 128, encoding = DW_ATE_complex_float>
! CHECK-DAG: #[[CMPX16:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "complex", sizeInBits = 256, encoding = DW_ATE_complex_float>

! CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "c4"{{.*}}type = #[[CMPX4]]>
! CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "c8"{{.*}}type = #[[CMPX8]]>
! CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "r"{{.*}}type = #[[CMPX16]]>
! CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "a"{{.*}}type = #[[CMPX4]]>
! CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "b"{{.*}}type = #[[CMPX8]]>
! CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "c"{{.*}}type = #[[CMPX16]]>

! LLVM-DAG: ![[C4:.*]] = !DIBasicType(name: "complex", size: 64, encoding: DW_ATE_complex_float)
! LLVM-DAG: ![[C8:.*]] = !DIBasicType(name: "complex", size: 128, encoding: DW_ATE_complex_float)
! LLVM-DAG: ![[C16:.*]] = !DIBasicType(name: "complex", size: 256, encoding: DW_ATE_complex_float)
! LLVM-DAG: !DILocalVariable(name: "c4"{{.*}}type: ![[C4]])
! LLVM-DAG: !DILocalVariable(name: "c8"{{.*}}type: ![[C8]])
! LLVM-DAG: !DILocalVariable(name: "r"{{.*}}type: ![[C16]])
! LLVM-DAG: !DILocalVariable(name: "a"{{.*}}type: ![[C4]])
! LLVM-DAG: !DILocalVariable(name: "b"{{.*}}type: ![[C8]])
! LLVM-DAG: !DILocalVariable(name: "c"{{.*}}type: ![[C16]])
