! RUN: bbc -emit-hlfir -fopenacc -fcuda -gpu=pinned %s -o - | FileCheck %s

subroutine declare_pinned_deallocate(a)
  real, allocatable :: a
  !$acc declare device_resident(a)
  deallocate(a)
end subroutine

! CHECK-LABEL: func.func @_QPdeclare_pinned_deallocate(
! CHECK-SAME: %[[A_ARG:.*]]: !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[A_ARG]]
! CHECK: cuf.deallocate %[[A]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>> {acc.declare_action = #acc.declare_action<preDealloc = @{{.*}}_acc_declare_pre_dealloc, postDealloc = @{{.*}}_acc_declare_post_dealloc>, data_attr = #cuf.cuda<pinned>} -> i32

! CHECK-LABEL: func.func private @{{.*}}_acc_declare_pre_dealloc(
! CHECK-SAME: %[[PRE_DEALLOC_ARG:.*]]: !fir.ref<!fir.box<!fir.heap<f32>>>)
! CHECK: acc.getdeviceptr varPtr(%[[PRE_DEALLOC_ARG]] : !fir.ref<!fir.box<!fir.heap<f32>>>)
! CHECK: acc.declare_exit
! CHECK: acc.delete

! CHECK-LABEL: func.func private @{{.*}}_acc_declare_post_dealloc(
! CHECK-SAME: %[[POST_DEALLOC_ARG:.*]]: !fir.ref<!fir.box<!fir.heap<f32>>>)
! CHECK: acc.getdeviceptr varPtr(%[[POST_DEALLOC_ARG]] : !fir.ref<!fir.box<!fir.heap<f32>>>){{.*}}implicit = true
! CHECK: acc.declare_exit
