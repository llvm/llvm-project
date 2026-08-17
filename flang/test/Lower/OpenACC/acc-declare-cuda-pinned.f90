! RUN: bbc -emit-hlfir -fopenacc -fcuda -gpu=pinned %s -o - | FileCheck %s

program declare_pinned
  real, allocatable :: a
  !$acc declare device_resident(a)
  allocate(a)
end program

! CHECK-LABEL: func.func @_QQmain()
! CHECK: %[[A_ALLOC:.*]] = cuf.alloc !fir.box<!fir.heap<f32>> {{.*}}data_attr = #cuf.cuda<pinned>
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[A_ALLOC]]
! CHECK: cuf.allocate %[[A]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>> {acc.declare_action = #acc.declare_action<postAlloc = @{{.*}}_acc_declare_post_alloc>, data_attr = #cuf.cuda<pinned>} -> i32
! CHECK: cuf.deallocate %[[A]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>> {acc.declare_action = #acc.declare_action<postDealloc = @{{.*}}_acc_declare_post_dealloc>, data_attr = #cuf.cuda<pinned>} -> i32

! CHECK-LABEL: func.func private @{{.*}}_acc_declare_post_alloc(
! CHECK-SAME: %[[POST_ALLOC_ARG:.*]]: !fir.ref<!fir.box<!fir.heap<f32>>>)
! CHECK: acc.declare_device_resident varPtr(%[[POST_ALLOC_ARG]] : !fir.ref<!fir.box<!fir.heap<f32>>>)
! CHECK: acc.declare_enter

! CHECK-LABEL: func.func private @{{.*}}_acc_declare_pre_dealloc(
! CHECK-SAME: %[[PRE_DEALLOC_ARG:.*]]: !fir.ref<!fir.box<!fir.heap<f32>>>)
! CHECK: acc.getdeviceptr varPtr(%[[PRE_DEALLOC_ARG]] : !fir.ref<!fir.box<!fir.heap<f32>>>)
! CHECK: acc.declare_exit
! CHECK: acc.delete

! CHECK-LABEL: func.func private @{{.*}}_acc_declare_post_dealloc(
! CHECK-SAME: %[[POST_DEALLOC_ARG:.*]]: !fir.ref<!fir.box<!fir.heap<f32>>>)
! CHECK: acc.getdeviceptr varPtr(%[[POST_DEALLOC_ARG]] : !fir.ref<!fir.box<!fir.heap<f32>>>){{.*}}implicit = true
! CHECK: acc.declare_exit
