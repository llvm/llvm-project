! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck  %s

! Test that complex return type is correctly represented in debug.
complex function fn(a)
    complex, intent(in) :: a
    fn = a
end function

! CHECK-DAG: ![[CMPLX:.*]] = !DIBasicType(name: "complex", size: 64, encoding: DW_ATE_complex_float)
! CHECK-DAG: ![[SR_TY:.*]] = !DISubroutineType(cc: DW_CC_normal, types: ![[TYPES:.*]])
! CHECK-DAG: ![[TYPES]] = !{![[CMPLX]], ![[CMPLX]]}
! CHECK-DAG: !DISubprogram(name: "fn"{{.*}}type: ![[SR_TY]]{{.*}})
