! Test lowering of ASYNCHRONOUS variables and IO statements.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

module test_async
contains
subroutine test(x, iounit, idvar, pending)
  real, asynchronous :: x(10)
  integer :: idvar, iounit
  logical :: pending
! CHECK-LABEL:   func.func @_QMtest_asyncPtest(
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}idvar
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}iounit
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %{{.*}}pending
! CHECK:           hlfir.declare %{{.*}}fir.var_attrs<asynchronous>{{.*}}x

  open(unit=iounit, asynchronous='yes')
! CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_14:.*]] = fir.call @_FortranAioBeginOpenUnit(%[[VAL_10]]
! CHECK:           %[[VAL_20:.*]] = fir.call @_FortranAioSetAsynchronous(%[[VAL_14]]
! CHECK:           %[[VAL_21:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_14]])

  write(unit=iounit,id=idvar, asynchronous='yes', fmt=*) x
! CHECK:           %[[VAL_22:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_26:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_22]],
! CHECK:           %[[VAL_32:.*]] = fir.call @_FortranAioSetAsynchronous(%[[VAL_26]],
! CHECK:           %[[VAL_36:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_26]],
! CHECK:           %[[VAL_37:.*]] = fir.call @_FortranAioGetAsynchronousId(%[[VAL_26]])
! CHECK:           fir.store %[[VAL_37]] to %[[VAL_4]]#1 : !fir.ref<i32>
! CHECK:           %[[VAL_38:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_26]])

  inquire(unit=iounit, id=idvar, pending=pending)
! CHECK:           %[[VAL_39:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_43:.*]] = fir.call @_FortranAioBeginInquireUnit(%[[VAL_39]],
! CHECK:           %[[VAL_44:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_46:.*]] = fir.convert %[[VAL_6]]#1 : (!fir.ref<!fir.logical<4>>) -> !fir.ref<i1>
! CHECK:           %[[VAL_47:.*]] = fir.call @_FortranAioInquirePendingId(%[[VAL_43]], %[[VAL_44]], %[[VAL_46]])
! CHECK:           %[[VAL_48:.*]] = fir.convert %[[VAL_6]]#1 : (!fir.ref<!fir.logical<4>>) -> !fir.ref<i1>
! CHECK:           %[[VAL_49:.*]] = fir.load %[[VAL_48]] : !fir.ref<i1>
! CHECK:           %[[VAL_50:.*]] = fir.convert %[[VAL_49]] : (i1) -> !fir.logical<4>
! CHECK:           fir.store %[[VAL_50]] to %[[VAL_6]]#1 : !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_51:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_43]])

  wait(unit=iounit, id=idvar)
! CHECK:           %[[VAL_52:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_53:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_57:.*]] = fir.call @_FortranAioBeginWait(%[[VAL_52]], %[[VAL_53]]
! CHECK:           %[[VAL_58:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_57]])
end subroutine
end module

  use test_async
  real :: x(10) = 1.
  integer :: iounit = 100
  integer :: idvar
  logical :: pending = .true.
  call test(x, iounit, idvar, pending)
  print *, idvar, pending
end
