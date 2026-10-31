! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s
! This test checks that, when casting to a volatile type, the corank attribute is forwarded.

module m 
  implicit none

  type :: t 
    character(len=4) :: c(2)
  end type 

contains

  subroutine display()
    class(t), volatile, allocatable, save :: cs[:]
    allocate(cs[*])
    print *, "[", cs%c(1), "][", cs%c(2), "]"
    deallocate(cs)
  end subroutine 

end module

! CHECK-LABEL: func.func @_QMmPdisplay()
! CHECK:   %[[VAL_1:.*]] = fir.address_of(@_QMmFdisplayEcs) : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMmTt{c:!fir.array<2x!fir.char<1,4>>}>>, corank:1>>
! CHECK:   %[[VAL_2:.*]] = fir.volatile_cast %[[VAL_1]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMmTt{c:!fir.array<2x!fir.char<1,4>>}>>, corank:1>>) -> !fir.ref<!fir.class<!fir.heap<!fir.type<_QMmTt{c:!fir.array<2x!fir.char<1,4>>}>>, volatile, corank:1>, volatile>


subroutine s()
  character(4), volatile, save :: c(2)[*]
  print *, "[", c(1), "][", c(2), "]"
end subroutine

! CHECK-LABEL: func.func @_QPs()
! CHECK:   %[[VAL_1:.*]] = fir.address_of(@_QFsEc) : !fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, corank:1>>
! CHECK:   %[[VAL_2:.*]] = fir.volatile_cast %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, corank:1>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.char<1,4>>>, volatile, corank:1>, volatile>
