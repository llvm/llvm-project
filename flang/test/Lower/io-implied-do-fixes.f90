! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s
! UNSUPPORTED: system-windows

! CHECK-LABEL: func @_QPido1
! CHECK: %[[IPTR_BOX_ADDR:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "iptr", uniq_name = "_QFido1Eiptr"}
! CHECK: %[[IPTR_DECL:.*]]:2 = hlfir.declare %[[IPTR_BOX_ADDR]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFido1Eiptr"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
! CHECK: %[[IPTR_BOX:.*]] = fir.load %[[IPTR_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK: %[[IPTR_ADDR:.*]] = fir.box_addr %[[IPTR_BOX]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK: %[[J_VAL_FINAL:.*]] = fir.do_loop %[[J_VAL:.*]] = %{{.*}} to %{{.*}} step %{{.*}} -> index {
! CHECK:   %[[J_VAL_CVT1:.*]] = fir.convert %[[J_VAL]] : (index) -> i32
! CHECK:   fir.store %[[J_VAL_CVT1]] to %[[IPTR_ADDR]] : !fir.ptr<i32>
! CHECK:   fir.result %[[J_VAL]] : index
! CHECK: }
! CHECK: %[[J_VAL_CVT2:.*]] = fir.convert %[[J_VAL_FINAL]] : (index) -> i32
! CHECK: fir.store %[[J_VAL_CVT2]] to %[[IPTR_ADDR]] : !fir.ptr<i32>
subroutine ido1
  integer, pointer :: iptr
  integer, target :: itgt
  iptr => itgt
  print *, (iptr,iptr=1,10)
end subroutine

! CHECK-LABEL: func @_QPido2
! CHECK: %[[IPTR_BOX_ADDR:.*]] = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "iptr", uniq_name = "_QFido2Eiptr"}
! CHECK: %[[IPTR_DECL:.*]]:2 = hlfir.declare %[[IPTR_BOX_ADDR]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFido2Eiptr"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
! CHECK: %[[IPTR_BOX:.*]] = fir.load %[[IPTR_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK: %[[IPTR_ADDR:.*]] = fir.box_addr %[[IPTR_BOX]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK: %[[J_VAL_FINAL:.*]] = fir.do_loop %[[J_VAL:.*]] = %{{.*}} to %{{.*}} step %{{.*}} -> index {
! CHECK:   %[[J_VAL_CVT1:.*]] = fir.convert %[[J_VAL]] : (index) -> i32
! CHECK:   fir.store %[[J_VAL_CVT1]] to %[[IPTR_ADDR]] : !fir.heap<i32>
! CHECK:   fir.result %[[J_VAL]] : index
! CHECK: }
! CHECK: %[[J_VAL_CVT2:.*]] = fir.convert %[[J_VAL_FINAL]] : (index) -> i32
! CHECK: fir.store %[[J_VAL_CVT2]] to %[[IPTR_ADDR]] : !fir.heap<i32>
subroutine ido2
  integer, allocatable :: iptr
  allocate(iptr)
  print *, (iptr,iptr=1,10)
end subroutine

! CHECK-LABEL: func @_QPido3
! CHECK: %[[J_BOX_ADDR:.*]] = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "j", uniq_name = "_QFido3Ej"}
! CHECK: %[[J_DECL:.*]]:2 = hlfir.declare %[[J_BOX_ADDR]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFido3Ej"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
! CHECK: %[[J_BOX:.*]] = fir.load %[[J_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK: %[[J_ADDR:.*]] = fir.box_addr %[[J_BOX]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK: %[[J_VAL_FINAL:.*]]:2 = fir.iterate_while (%[[J_VAL:.*]] = %{{.*}} to %{{.*}} step %{{.*}}) and (%[[OK:.*]] = {{.*}}) -> (index, i1) {
! CHECK:   %[[J_VAL_CVT1:.*]] = fir.convert %[[J_VAL]] : (index) -> i32
! CHECK:   fir.store %[[J_VAL_CVT1]] to %[[J_ADDR]] : !fir.heap<i32>
! CHECK:   %{{.*}} = fir.if %[[OK]] -> (i1) {
! CHECK:     %[[RES:.*]] = fir.call @_FortranAioOutputInteger32({{.*}}) {{.*}}: (!fir.ref<i8>, i32) -> i1
! CHECK:     fir.result %[[RES]] : i1
! CHECK:   } else {
! CHECK:     fir.result %{{.*}} : i1
! CHECK:   }
! CHECK:   %[[J_VAL_INC:.*]] = arith.addi %[[J_VAL]], %{{[^ ]*}} overflow<nsw> : index
! CHECK:   %[[J_VAL_NEXT:.*]] = arith.select %{{.*}}, %[[J_VAL_INC]], %[[J_VAL]] : index
! CHECK:   fir.result %[[J_VAL_NEXT]], %{{.*}} : index, i1
! CHECK: }
! CHECK: %[[J_VAL_CVT2:.*]] = fir.convert %[[J_VAL_FINAL]]#0 : (index) -> i32
! CHECK: fir.store %[[J_VAL_CVT2]] to %[[J_ADDR]] : !fir.heap<i32>
subroutine ido3
  integer, allocatable :: j
  allocate(j)
  write(*,*,err=404) (j,j=1,10)
404 continue
end subroutine
