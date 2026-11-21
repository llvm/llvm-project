! This test checks lowering of OpenMP loops with pointer or allocatable loop index variable

!RUN: bbc -emit-hlfir -fopenmp %s -o - 2>&1 | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - 2>&1 | FileCheck %s

!CHECK-LABEL: QQmain
program loop_var
  integer, pointer :: ip1, ip2
  integer, allocatable :: ia1

!CHECK:  %[[IA1:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFEia1") fortran_attrs<allocatable>
!CHECK:  omp.wsloop private(@_QFEip1_private_box_ptr_i32 %{{.*}}#0 -> %[[IP1_PVT:.*]], @_QFEip2_private_box_ptr_i32 %{{.*}}#0 -> %[[IP2_PVT:.*]] : !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK:  omp.loop_nest (%[[IP1_INDX:.*]], %[[IP2_INDX:.*]]) : i32 = ({{.*}}) to ({{.*}}) inclusive step ({{.*}})
!CHECK:    %[[IP1_PVT_DECL:.*]]:2 = hlfir.declare %[[IP1_PVT]] uniq_name("_QFEip1") fortran_attrs<pointer> : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK:    %[[IP2_PVT_DECL:.*]]:2 = hlfir.declare %[[IP2_PVT]] uniq_name("_QFEip2") fortran_attrs<pointer> : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK:    %[[IP1_BOX:.*]] = fir.load %[[IP1_PVT_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK:    %[[IP1_ADDR:.*]] = fir.box_addr %[[IP1_BOX]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK:    hlfir.assign %[[IP1_INDX]] to %[[IP1_ADDR]] : i32, !fir.ptr<i32>
!CHECK:    %[[IP2_BOX:.*]] = fir.load %[[IP2_PVT_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK:    %[[IP2_ADDR:.*]] = fir.box_addr %[[IP2_BOX]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK:    hlfir.assign %[[IP2_INDX]] to %[[IP2_ADDR]] : i32, !fir.ptr<i32>
!CHECK:    omp.yield
  !$omp do collapse(2)
  do ip1 = 1, 10
    do ip2 = 1, 20
    end do
  end do
  !$omp end do

!CHECK:    omp.simd linear(@_QFEia1_linear_box_heap_i32 %[[IA1]]#0 -> %[[IA1_PVT:.*]] : !fir.ref<!fir.box<!fir.heap<i32>>> = %{{.*}} : i32) linear_var_types([i32])
!CHECK:      omp.loop_nest (%[[IA1_INDX:.*]]) : i32 = ({{.*}}) to ({{.*}}) inclusive step ({{.*}})
!CHECK:        %[[IA1_PVT_DECL:.*]]:2 = hlfir.declare %[[IA1_PVT]] uniq_name("_QFEia1") fortran_attrs<allocatable>
!CHECK:        %[[IA1_BOX:.*]] = fir.load %[[IA1_PVT_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
!CHECK:        %[[IA1_ADDR:.*]] = fir.box_addr %[[IA1_BOX]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
!CHECK:        hlfir.assign %[[IA1_INDX]] to %[[IA1_ADDR]] : i32, !fir.heap<i32>
!CHECK:        omp.yield
  !$omp simd
  do ia1 = 1, 10
  end do
  !$omp end simd
end program
