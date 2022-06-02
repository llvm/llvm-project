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

  ! CHECK: %[[len:.*]] = arith.addi %[[a]]#1, %[[b]]#1
  ! CHECK: %[[temp:.*]] = fir.alloca !fir.char<1,?>(%[[len]] : index)

  ! CHECK-DAG: %[[c1:.*]] = arith.constant 1
  ! CHECK-DAG: %[[a2:.*]] = fir.convert %[[a]]#1
  ! CHECK: %[[count:.*]] = arith.muli %[[c1]], %[[a2]]
  ! CHECK-DAG: constant false
  ! CHECK-DAG: %[[to:.*]] = fir.convert %[[temp]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK-DAG: %[[from:.*]] = fir.convert %[[a]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0.p0.i64(%[[to]], %[[from]], %[[count]], %false) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()

  ! CHECK: %[[c1_0:.*]] = arith.constant 1
  ! CHECK: %[[count2:.*]] = arith.subi %[[len]], %[[c1_0]]
  ! CHECK: fir.do_loop %[[index2:.*]] = %[[a]]#1 to %[[count2]] step %[[c1_0]] {
    ! CHECK: %[[b_index:.*]] = arith.subi %[[index2]], %[[a]]#1
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
