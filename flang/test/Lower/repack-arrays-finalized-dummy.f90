! RUN: bbc -emit-hlfir -frepack-arrays %s -o - -I nowhere | FileCheck --check-prefixes=CHECK %s

! Check that the original array is copied on entry to the subroutine
! before it is being finalized, otherwise the finalization routine
! may read the uninitialized temporary.
! Verify that fir.pack_array does not have no_copy attribute.

module m
  type t
   contains
     final :: my_final
  end type t
  interface
     subroutine my_final(x)
       type(t) :: x(:)
     end subroutine my_final
  end interface
contains
! CHECK-LABEL:   func.func @_QMmPtest(
! CHECK-SAME:                         %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.class<!fir.array<?x!fir.type<_QMmTt>>> {fir.bindc_name = "x"}) {
  subroutine test(x)
    class(t), intent(out) :: x(:)
! CHECK:           %[[VAL_2:.*]] = fir.pack_array %[[VAL_0]] heap whole : (!fir.class<!fir.array<?x!fir.type<_QMmTt>>>) -> !fir.class<!fir.array<?x!fir.type<_QMmTt>>>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]]
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]]#1
! CHECK:           fir.call @_FortranADestroy(%[[VAL_4]]) fastmath<contract> : (!fir.box<none>) -> ()
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_3]]#1
! CHECK:           fir.call @_FortranAInitialize(%[[VAL_7]]
! CHECK:           fir.unpack_array %[[VAL_2]] to %[[VAL_0]] heap : !fir.class<!fir.array<?x!fir.type<_QMmTt>>>
  end subroutine test
end module m
