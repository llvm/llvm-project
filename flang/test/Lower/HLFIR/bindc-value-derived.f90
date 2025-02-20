! Test lowering of derived types passed with VALUE attribute in BIND(C)
! interface. They are passed as fir.type<T> value. The actual C struct
! passing ABI is done in code generation according to the target.

! RUN: bbc -emit-hlfir -o - -I nw %s 2>&1 | FileCheck %s

module bindc_byval
  type, bind(c) :: t
    integer :: i
  end type
contains
  subroutine test(x) bind(c)
    type(t), value :: x
    call use_it(x%i)
  end subroutine
! CHECK-LABEL:   func.func @test(
! CHECK-SAME:                    %[[VAL_0:.*]]: !fir.type<_QMbindc_byvalTt{{[<]?}}{i:i32}{{[>]?}}>
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.type<_QMbindc_byvalTt{{[<]?}}{i:i32}{{[>]?}}>
! CHECK:           fir.store %[[VAL_0]] to %[[VAL_1]] : !fir.ref<!fir.type<_QMbindc_byvalTt{{[<]?}}{i:i32}{{[>]?}}>>
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{[0-9]}} {fortran_attrs = #fir.var_attrs<value>, uniq_name = "_QMbindc_byvalFtestEx"} : (!fir.ref<!fir.type<_QMbindc_byvalTt{{[<]?}}{i:i32}{{[>]?}}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMbindc_byvalTt{{[<]?}}{i:i32}{{[>]?}}>>, !fir.ref<!fir.type<_QMbindc_byvalTt{{[<]?}}{i:i32}{{[>]?}}>>)
! CHECK:           %[[VAL_3:.*]] = hlfir.designate %[[VAL_2]]#0{"i"}   : (!fir.ref<!fir.type<_QMbindc_byvalTt{{[<]?}}{i:i32}{{[>]?}}>>) -> !fir.ref<i32>
! CHECK:           fir.call @_QPuse_it(%[[VAL_3]]) fastmath<contract> : (!fir.ref<i32>) -> ()
! CHECK:           return
! CHECK:         }

  subroutine call_it(x)
    type(t) x
    call test(x)
  end subroutine
! CHECK-LABEL:   func.func @_QMbindc_byvalPcall_it(
! CHECK-SAME:                                      %[[VAL_0:.*]]: !fir.ref<!fir.type<_QMbindc_byvalTt{{[<]?}}{i:i32}{{[>]?}}>>
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]}} {uniq_name = "_QMbindc_byvalFcall_itEx"} : (!fir.ref<!fir.type<_QMbindc_byvalTt{{[<]?}}{i:i32}{{[>]?}}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMbindc_byvalTt{{[<]?}}{i:i32}{{[>]?}}>>, !fir.ref<!fir.type<_QMbindc_byvalTt{{[<]?}}{i:i32}{{[>]?}}>>)
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_1]]#1 : !fir.ref<!fir.type<_QMbindc_byvalTt{{[<]?}}{i:i32}{{[>]?}}>>
! CHECK:           fir.call @test(%[[VAL_2]]) proc_attrs<bind_c> fastmath<contract> : (!fir.type<_QMbindc_byvalTt{{[<]?}}{i:i32}{{[>]?}}>) -> ()
! CHECK:           return
! CHECK:         }
end module
