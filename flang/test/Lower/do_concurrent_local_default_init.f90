! Test default initialization of DO CONCURRENT LOCAL() entities.
! RUN: bbc -emit-hlfir --enable-delayed-privatization-staging=true -I nowhere -o - %s | FileCheck %s

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

! CHECK: fir.local {type = local} @[[DEFAULT_INIT_LOCALIZER:.*test_default_init.*]] : !fir.type<{{.*}}test_default_initTt{i:i32}> init {
! CHECK-NEXT: ^{{.*}}(%{{.*}}: !{{.*}}, %[[LOCAL_ARG:.*]]: !fir.ref<!fir.type<_QFtest_default_initTt{i:i32}>>):
! CHECK-NEXT:   %[[LOCAL_ARG_BOX:.*]] = fir.embox %[[LOCAL_ARG]]
! CHECK:        %[[LOCAL_ARG_BOX_CVT:.*]] = fir.convert %[[LOCAL_ARG_BOX]]
! CHECK:        fir.call @_FortranAInitialize(%[[LOCAL_ARG_BOX_CVT]], {{.*}})
! CHECK-NEXT:   fir.yield(%[[LOCAL_ARG]] : {{.*}})
! CHECK-NEXT: }

! CHECK: fir.local {type = local} @[[PTR_LOCALIZER:.*test_ptrEp_private_box.*]] : !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>> init {
! CHECK-NEXT: ^{{.*}}(%[[ORIG_ARG:.*]]: !{{.*}}, %[[LOCAL_ARG:.*]]: !{{.*}}):
! CHECK-NEXT:   %[[ORIG_ARG_LD:.*]] = fir.load %[[ORIG_ARG]]
! CHECK-NEXT:   %[[ELEM_SIZE:.*]] = fir.box_elesize %[[ORIG_ARG_LD]]
! CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
! CHECK-NEXT:   %[[SHAPE:.*]] = fir.shape %[[C0]]
! CHECK-NEXT:   %[[ZERO_BITS:.*]] = fir.zero_bits
! CHECK-NEXT:   %[[LOCAL_BOX:.*]] = fir.embox %[[ZERO_BITS]](%[[SHAPE]]) typeparams %[[ELEM_SIZE]]
! CHECK-NEXT:   fir.store %[[LOCAL_BOX]] to %[[LOCAL_ARG]]
! CHECK-NEXT:   fir.yield(%[[LOCAL_ARG]] : {{.*}})
! CHECK-NEXT: }

! CHECK-LABEL:   func.func @_QPtest_ptr(
! CHECK-SAME:                           %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>> {fir.bindc_name = "p"}) {
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:           %[[VAL_7:.*]] = fir.box_elesize %[[VAL_6]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>) -> index
! CHECK:           fir.do_concurrent.loop {{.*}} local(@[[PTR_LOCALIZER]] %{{.*}}#0 -> %[[LOCAL_ARG:.*]] : {{.*}})
! CHECK:             %[[VAL_21:.*]]:2 = hlfir.declare %[[LOCAL_ARG]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_ptrEp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>)
! CHECK:             fir.call @_QPtakes_ptr(%[[VAL_21]]#0) proc_attrs<pure> fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>) -> ()
! CHECK:           }
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_default_init(
! CHECK:           fir.do_concurrent.loop {{.*}} local(@[[DEFAULT_INIT_LOCALIZER]] %{{.*}}#0 -> %[[LOCAL_ARG:.*]] : {{.*}})
! CHECK:             %[[VAL_33:.*]]:2 = hlfir.declare %[[LOCAL_ARG]] {uniq_name = "_QFtest_default_initEa"} : (!fir.ref<!fir.type<_QFtest_default_initTt{i:i32}>>) -> (!fir.ref<!fir.type<_QFtest_default_initTt{i:i32}>>, !fir.ref<!fir.type<_QFtest_default_initTt{i:i32}>>)
! CHECK:           }
