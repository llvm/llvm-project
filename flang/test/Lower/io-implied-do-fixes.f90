! RUN: bbc --use-desc-for-alloc=false -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: bbc --use-desc-for-alloc=false -emit-fir -hlfir=false -fwrapv %s -o - | FileCheck %s --check-prefix=NO-NSW
! UNSUPPORTED: system-windows

! NO-NSW-NOT: overflow<nsw>

! CHECK-LABEL: func @_QPido1
! CHECK: %[[J_REF_ADDR:.*]] = fir.alloca !fir.ptr<i32> {uniq_name = "_QFido1Eiptr.addr"}
! CHECK: %[[J_ADDR:.*]] = fir.load %[[J_REF_ADDR]] : !fir.ref<!fir.ptr<i32>>
! CHECK: %[[J_VAL_FINAL:.*]] = fir.do_loop %[[J_VAL:.*]] = %{{.*}} to %{{.*}} step %{{.*}} -> index {
! CHECK:   %[[J_VAL_CVT1:.*]] = fir.convert %[[J_VAL]] : (index) -> i32
! CHECK:   fir.store %[[J_VAL_CVT1]] to %[[J_ADDR]] : !fir.ptr<i32>
! CHECK:   %[[J_VAL_NEXT:.*]] = arith.addi %[[J_VAL]], %{{[^ ]*}} overflow<nsw> : index
! CHECK:   fir.result %[[J_VAL_NEXT]] : index
! CHECK: }
! CHECK: %[[J_VAL_CVT2:.*]] = fir.convert %[[J_VAL_FINAL]] : (index) -> i32
! CHECK: fir.store %[[J_VAL_CVT2]] to %[[J_ADDR]] : !fir.ptr<i32>
subroutine ido1
  integer, pointer :: iptr
  integer, target :: itgt
  iptr => itgt
  print *, (iptr,iptr=1,10)
end subroutine

! CHECK-LABEL: func @_QPido2
! CHECK: %[[J_REF_ADDR:.*]] = fir.alloca !fir.heap<i32> {uniq_name = "_QFido2Eiptr.addr"}
! CHECK: %[[J_ADDR:.*]] = fir.load %[[J_REF_ADDR]] : !fir.ref<!fir.heap<i32>>
! CHECK: %[[J_VAL_FINAL:.*]] = fir.do_loop %[[J_VAL:.*]] = %{{.*}} to %{{.*}} step %{{.*}} -> index {
! CHECK: %[[J_VAL_CVT1:.*]] = fir.convert %[[J_VAL]] : (index) -> i32
! CHECK: fir.store %[[J_VAL_CVT1]] to %[[J_ADDR]] : !fir.heap<i32>
! CHECK: %[[J_VAL_NEXT:.*]] = arith.addi %[[J_VAL]], %{{[^ ]*}} overflow<nsw> : index
! CHECK: fir.result %[[J_VAL_NEXT]] : index
! CHECK: }
! CHECK: %[[J_VAL_CVT2:.*]] = fir.convert %[[J_VAL_FINAL]] : (index) -> i32
! CHECK: fir.store %[[J_VAL_CVT2]] to %[[J_ADDR]] : !fir.heap<i32>
subroutine ido2
  integer, allocatable :: iptr
  allocate(iptr)
  print *, (iptr,iptr=1,10)
end subroutine

! CHECK-LABEL: func @_QPido3
! CHECK:  %[[J_REF_ADDR:.*]] = fir.alloca !fir.heap<i32> {uniq_name = "_QFido3Ej.addr"}
! CHECK:  %[[J_ADDR:.*]] = fir.load %[[J_REF_ADDR]] : !fir.ref<!fir.heap<i32>>
! CHECK:  %[[J_VAL_FINAL:.*]]:2 = fir.iterate_while (%[[J_VAL:.*]] = %{{.*}} to %{{.*}} step %{{.*}}) and (%[[OK:.*]] = {{.*}}) -> (index, i1) {
! CHECK:    %[[J_VAL_CVT1:.*]] = fir.convert %[[J_VAL]] : (index) -> i32
! CHECK:    fir.store %[[J_VAL_CVT1]] to %[[J_ADDR]] : !fir.heap<i32>
! CHECK:    %[[RES:.*]] = fir.if %[[OK]] -> (i1) {
! CHECK:    }
! CHECK:   %[[J_VAL_INC:.*]] = arith.addi %[[J_VAL]], %{{[^ ]*}} overflow<nsw> : index
! CHECK:   %[[J_VAL_NEXT:.*]] = arith.select %[[RES]], %[[J_VAL_INC]], %[[J_VAL]] : index
! CHECK:   fir.result %[[J_VAL_NEXT]], %[[RES]] : index, i1
! CHECK:  }
! CHECK:  %[[J_VAL_CVT2:.*]] = fir.convert %[[J_VAL_FINAL]]#0 : (index) -> i32
! CHECK:  fir.store %[[J_VAL_CVT2]] to %[[J_ADDR]] : !fir.heap<i32
subroutine ido3
  integer, allocatable :: j
  allocate(j)
  write(*,*,err=404) (j,j=1,10)
404 continue
end subroutine
