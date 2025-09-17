! RUN: bbc -emit-hlfir %s -o - | FileCheck %s
! Test lowering of extension of SOURCE allocation (non deferred length
! of character allocate-object need not to match the SOURCE length, truncation
! and padding are performed instead as in assignments).

subroutine test()
! CHECK-LABEL:   func.func @_QPtest() {
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}} {{.*}}Ec_deferred
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %{{.*}} typeparams %[[VAL_6:.*]] {{.*}}Ec_longer
! CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %{{.*}} typeparams %[[VAL_11:.*]] {{.*}}Ec_shorter
! CHECK:           %[[VAL_17:.*]]:2 = hlfir.declare %{{.*}} typeparams %[[VAL_16:.*]] {{{.*}}Ec_source
  character(5) :: c_source = "hello"
  character(2), allocatable :: c_shorter
  character(:), allocatable :: c_deferred
  character(7), allocatable :: c_longer
! CHECK-DAG:           %[[VAL_18:.*]] = arith.constant false
! CHECK:           %[[VAL_22:.*]] = fir.embox %[[VAL_17]]#0 : (!fir.ref<!fir.char<1,5>>) -> !fir.box<!fir.char<1,5>>

! CHECK-DAG:           %[[VAL_23:.*]] = fir.convert %[[VAL_14]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,2>>>>) -> !fir.ref<!fir.box<none>>
! CHECK-DAG:           %[[VAL_24:.*]] = fir.convert %[[VAL_22]] : (!fir.box<!fir.char<1,5>>) -> !fir.box<none>
! CHECK:           %[[VAL_26:.*]] = fir.call @_FortranAAllocatableAllocateSource(%[[VAL_23]], %[[VAL_24]], %[[VAL_18]]

! CHECK-DAG:           %[[VAL_27:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK-DAG:           %[[VAL_28:.*]] = fir.convert %[[VAL_16]] : (index) -> i64
! CHECK-DAG:           %[[VAL_29:.*]] = arith.constant 1 : i32
! CHECK-DAG:           %[[VAL_30:.*]] = arith.constant 0 : i32
! CHECK-DAG:           %[[VAL_31:.*]] = arith.constant 0 : i32
! CHECK:           fir.call @_FortranAAllocatableInitCharacterForAllocate(%[[VAL_27]], %[[VAL_28]], %[[VAL_29]], %[[VAL_30]], %[[VAL_31]]
! CHECK-DAG:           %[[VAL_33:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK-DAG:           %[[VAL_34:.*]] = fir.convert %[[VAL_22]] : (!fir.box<!fir.char<1,5>>) -> !fir.box<none>
! CHECK:           %[[VAL_36:.*]] = fir.call @_FortranAAllocatableAllocateSource(%[[VAL_33]], %[[VAL_34]], %[[VAL_18]],

! CHECK-NOT: AllocatableInitCharacterForAllocate
! CHECK-DAG:           %[[VAL_37:.*]] = fir.convert %[[VAL_9]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,7>>>>) -> !fir.ref<!fir.box<none>>
! CHECK-DAG:           %[[VAL_38:.*]] = fir.convert %[[VAL_22]] : (!fir.box<!fir.char<1,5>>) -> !fir.box<none>
! CHECK:           %[[VAL_40:.*]] = fir.call @_FortranAAllocatableAllocateSource(%[[VAL_37]], %[[VAL_38]], %[[VAL_18]],
  allocate(c_shorter, c_deferred, c_longer, source=c_source)

! Expect at runtime:
! ZZheZZ
! ZZhelloZZ
! ZZhello  ZZ
  write(*,"('ZZ',A,'ZZ')") c_shorter
  write(*,"('ZZ',A,'ZZ')") c_deferred
  write(*,"('ZZ',A,'ZZ')") c_longer
end subroutine

  call test()
end
