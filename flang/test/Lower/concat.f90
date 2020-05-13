! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test character scalar concatenation lowering

! CHECK-LABEL: concat_1
subroutine concat_1(a, b)
  character(*) :: a, b
  ! CHECK: call @{{.*}}BeginExternalListOutput
  ! CHECK-DAG: %[[a:.*]]:2 = fir.unboxchar %arg0
  ! CHECK-DAG: %[[b:.*]]:2 = fir.unboxchar %arg1

  print *, a // b
  ! Concatenation

  ! CHECK: %[[len:.*]] = addi %[[a]]#1, %[[b]]#1
  ! CHECK: %[[temp:.*]] = fir.alloca !fir.char<1>, %[[len]]

  ! CHECK-DAG: %[[c0:.*]] = constant 0
  ! CHECK-DAG: %[[c1:.*]] = constant 1
  ! CHECK-DAG: %[[count:.*]] = subi %[[a]]#1, %[[c1]]
  ! CHECK: fir.do_loop %[[index:.*]] = %[[c0]] to %[[count]] step %[[c1]] {
    ! CHECK: %[[a_addr:.*]] = fir.coordinate_of %[[a]]#0, %[[index]]
    ! CHECK: %[[a_elt:.*]] = fir.load %[[a_addr]]
    ! CHECK: %[[temp_addr:.*]] = fir.coordinate_of %[[temp]], %[[index]]
    ! CHECK: fir.store %[[a_elt]] to %[[temp_addr]]
  ! CHECK: }

  ! CHECK: %[[c1_0:.*]] = constant 1
  ! CHECK: %[[count2:.*]] = subi %[[len]], %[[c1_0]]
  ! CHECK: fir.do_loop %[[index2:.*]] = %[[a]]#1 to %[[count2]] step %[[c1_0]] {
    ! CHECK: %[[b_index:.*]] = subi %[[index]], %[[a]]#1
    ! CHECK: %[[b_addr:.*]] = fir.coordinate_of %[[b]]#0, %[[b_index]]
    ! CHECK: %[[b_elt:.*]] = fir.load %[[b_addr]]
    ! CHECK: %[[temp_addr2:.*]] = fir.coordinate_of %[[temp]], %[[index2]]
    ! CHECK: fir.store %[[b_elt]] to %[[temp_addr2]]
  ! CHECK: }

  ! CHECK: %[[embox_temp:.*]] = fir.emboxchar %[[temp]], %[[len]]

  ! IO runtime call
  ! CHECK: %[[result:.*]]:2 = fir.unboxchar %[[embox_temp]]
  ! CHECK-DAG: %[[raddr:.*]] = fir.convert %[[result]]#0
  ! CHECK-DAG: %[[rlen:.*]] = fir.convert %[[result]]#1
  ! CHECK: call @{{.*}}OutputAscii(%{{.*}}, %[[raddr]], %[[rlen]])
end subroutine
