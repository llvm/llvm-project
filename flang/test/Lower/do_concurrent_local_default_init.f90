! Test default initialization of DO CONCURRENT LOCAL() entities.
! RUN: bbc -emit-hlfir -I nowhere -o - %s | FileCheck %s

subroutine test_ptr(p)
  interface
    pure subroutine takes_ptr(p)
      character(*), intent(in), pointer :: p(:)
    end subroutine
  end interface
  character(*), pointer :: p(:)
  integer :: i
  do concurrent (i=1:10) local(p)
    call takes_ptr(p)
  end do
end subroutine

subroutine test_default_init()
  type t
    integer :: i = 2
  end type
  integer :: i, res(4)
  type(t) :: a
  do concurrent (i=1:4) local(a)
    res(i) = a%i
  end do
  call something(res)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_ptr(
! CHECK-SAME:                           %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>> {fir.bindc_name = "p"}) {
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:           %[[VAL_7:.*]] = fir.box_elesize %[[VAL_6]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>) -> index
! CHECK:           fir.do_loop
! CHECK:             %[[VAL_16:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>> {bindc_name = "p", pinned, uniq_name = "_QFtest_ptrEp"}
! CHECK:             %[[VAL_17:.*]] = fir.zero_bits !fir.ptr<!fir.array<?x!fir.char<1,?>>>
! CHECK:             %[[VAL_18:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_19:.*]] = fir.shape %[[VAL_18]] : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_20:.*]] = fir.embox %[[VAL_17]](%[[VAL_19]]) typeparams %[[VAL_7]] : (!fir.ptr<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
! CHECK:             fir.store %[[VAL_20]] to %[[VAL_16]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:             %[[VAL_21:.*]]:2 = hlfir.declare %[[VAL_16]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_ptrEp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>)
! CHECK:             fir.call @_QPtakes_ptr(%[[VAL_21]]#0) proc_attrs<pure> fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>) -> ()
! CHECK:           }
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_default_init(
! CHECK:           fir.do_loop
! CHECK:             %[[VAL_26:.*]] = fir.alloca !fir.type<_QFtest_default_initTt{i:i32}> {bindc_name = "a", pinned, uniq_name = "_QFtest_default_initEa"}
! CHECK:             %[[VAL_27:.*]] = fir.embox %[[VAL_26]] : (!fir.ref<!fir.type<_QFtest_default_initTt{i:i32}>>) -> !fir.box<!fir.type<_QFtest_default_initTt{i:i32}>>
! CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_27]] : (!fir.box<!fir.type<_QFtest_default_initTt{i:i32}>>) -> !fir.box<none>
! CHECK:             fir.call @_FortranAInitialize(%[[VAL_30]], {{.*}}
! CHECK:             %[[VAL_33:.*]]:2 = hlfir.declare %[[VAL_26]] {uniq_name = "_QFtest_default_initEa"} : (!fir.ref<!fir.type<_QFtest_default_initTt{i:i32}>>) -> (!fir.ref<!fir.type<_QFtest_default_initTt{i:i32}>>, !fir.ref<!fir.type<_QFtest_default_initTt{i:i32}>>)
! CHECK:           }
