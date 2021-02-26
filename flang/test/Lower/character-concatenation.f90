! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test character scalar concatenation lowering

! CHECK-LABEL: concat_1
subroutine concat_1(a, b)
  ! CHECK-DAG: %[[a:.*]]:2 = fir.unboxchar %arg0
  ! CHECK-DAG: %[[b:.*]]:2 = fir.unboxchar %arg1
  character(*) :: a, b

  ! CHECK: call @{{.*}}BeginExternalListOutput
  print *, a // b
  ! Concatenation

  ! CHECK: %[[len:.*]] = addi %[[a]]#1, %[[b]]#1
  ! CHECK: %[[temp:.*]] = fir.alloca !fir.char<1,?>, %[[len]]

  ! CHECK-DAG: %[[c0:.*]] = constant 0
  ! CHECK-DAG: %[[c1:.*]] = constant 1
  ! CHECK-DAG: %[[count:.*]] = subi %[[a]]#1, %[[c1]]
  ! CHECK: fir.do_loop %[[index:.*]] = %[[c0]] to %[[count]] step %[[c1]] {
    ! CHECK: %[[a_cast:.*]] = fir.convert %[[a]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
    ! CHECK: %[[a_addr:.*]] = fir.coordinate_of %[[a_cast]], %[[index]]
    ! CHECK-DAG: %[[a_elt:.*]] = fir.load %[[a_addr]]
    ! CHECK: %[[temp_cast:.*]] = fir.convert %[[temp]]
    ! CHECK: %[[temp_addr:.*]] = fir.coordinate_of %[[temp_cast]], %[[index]]
    ! CHECK: fir.store %[[a_elt]] to %[[temp_addr]]
  ! CHECK: }

  ! CHECK: %[[c1_0:.*]] = constant 1
  ! CHECK: %[[count2:.*]] = subi %[[len]], %[[c1_0]]
  ! CHECK: fir.do_loop %[[index2:.*]] = %[[a]]#1 to %[[count2]] step %[[c1_0]] {
    ! CHECK: %[[b_index:.*]] = subi %[[index]], %[[a]]#1
    ! CHECK: %[[b_cast:.*]] = fir.convert %[[b]]#0
    ! CHECK: %[[b_addr:.*]] = fir.coordinate_of %[[b_cast]], %[[b_index]]
    ! CHECK-DAG: %[[b_elt:.*]] = fir.load %[[b_addr]]
    ! CHECK: %[[temp_cast2:.*]] = fir.convert %[[temp]]
    ! CHECK: %[[temp_addr2:.*]] = fir.coordinate_of %[[temp_cast2]], %[[index2]]
    ! CHECK: fir.store %[[b_elt]] to %[[temp_addr2]]
  ! CHECK: }

  ! IO runtime call
  ! CHECK-DAG: %[[raddr:.*]] = fir.convert %[[temp]]
  ! CHECK-DAG: %[[rlen:.*]] = fir.convert %[[len]]
  ! CHECK: call @{{.*}}OutputAscii(%{{.*}}, %[[raddr]], %[[rlen]])
end subroutine
