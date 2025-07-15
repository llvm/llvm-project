! RUN: %flang_fc1 -emit-hlfir -mmlir --enable-delayed-privatization-staging=true -o - %s | FileCheck %s

subroutine do_concurrent_with_locality_specs
  implicit none
  integer :: i, local_var, local_init_var

  do concurrent (i=1:10) local(local_var) local_init(local_init_var)
    if (i < 5) then
      local_var = 42
    else 
      local_init_var = 84
    end if
  end do
end subroutine

! CHECK:         fir.local {type = local_init} @[[LOCAL_INIT_SYM:.*]] : i32 copy {
! CHECK:         ^bb0(%[[ORIG_VAL:.*]]: !fir.ref<i32>, %[[LOCAL_VAL:.*]]: !fir.ref<i32>):
! CHECK:           %[[ORIG_VAL_LD:.*]] = fir.load %[[ORIG_VAL]] : !fir.ref<i32>
! CHECK:           hlfir.assign %[[ORIG_VAL_LD]] to %[[LOCAL_VAL]] : i32, !fir.ref<i32>
! CHECK:           fir.yield(%[[LOCAL_VAL]] : !fir.ref<i32>)
! CHECK:         }

! CHECK:         fir.local {type = local} @[[LOCAL_SYM:.*]] : i32

! CHECK-LABEL:   func.func @_QPdo_concurrent_with_locality_specs() {
! CHECK:           %[[ORIG_LOCAL_INIT_ALLOC:.*]] = fir.alloca i32 {bindc_name = "local_init_var", {{.*}}}
! CHECK:           %[[ORIG_LOCAL_INIT_DECL:.*]]:2 = hlfir.declare %[[ORIG_LOCAL_INIT_ALLOC]]

! CHECK:           %[[ORIG_LOCAL_ALLOC:.*]] = fir.alloca i32 {bindc_name = "local_var", {{.*}}}
! CHECK:           %[[ORIG_LOCAL_DECL:.*]]:2 = hlfir.declare %[[ORIG_LOCAL_ALLOC]]

! CHECK:           fir.do_concurrent {
! CHECK:             %[[IV_DECL:.*]]:2 = hlfir.declare %{{.*}}

! CHECK:             fir.do_concurrent.loop (%{{.*}}) = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) local(@[[LOCAL_SYM]] %[[ORIG_LOCAL_DECL]]#0 -> %[[LOCAL_ARG:.*]], @[[LOCAL_INIT_SYM]] %[[ORIG_LOCAL_INIT_DECL]]#0 -> %[[LOCAL_INIT_ARG:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK:               %[[LOCAL_DECL:.*]]:2 = hlfir.declare %[[LOCAL_ARG]]
! CHECK:               %[[LOCAL_INIT_DECL:.*]]:2 = hlfir.declare %[[LOCAL_INIT_ARG]]

! CHECK:               fir.if %{{.*}} {
! CHECK:                 %[[C42:.*]] = arith.constant 42 : i32
! CHECK:                 hlfir.assign %[[C42]] to %[[LOCAL_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:               } else {
! CHECK:                 %[[C84:.*]] = arith.constant 84 : i32
! CHECK:                 hlfir.assign %[[C84]] to %[[LOCAL_INIT_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:               }
! CHECK:             }
! CHECK:           }
! CHECK:           return
! CHECK:         }
