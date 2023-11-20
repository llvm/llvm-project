! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

  ! Make sure we use array values for subscripts that are arrays on the lhs so
  ! that copy-in/copy-out works correctly.
  integer :: a(4,4)
  forall(i=1:4,j=1:4)
    a(a(i,j),a(j,i)) = j - i*100
  end forall
end

! CHECK-LABEL: func @_QQmain
! CHECK: %[[a:.*]] = fir.address_of(@_QFEa) : !fir.ref<!fir.array<4x4xi32>>
! CHECK: %[[a1:.*]] = fir.array_load %[[a]](%{{.*}}) : (!fir.ref<!fir.array<4x4xi32>>, !fir.shape<2>) -> !fir.array<4x4xi32>
! CHECK: %[[a2:.*]] = fir.array_load %[[a]](%{{.*}}) : (!fir.ref<!fir.array<4x4xi32>>, !fir.shape<2>) -> !fir.array<4x4xi32>
! CHECK: %[[a3:.*]] = fir.array_load %[[a]](%{{.*}}) : (!fir.ref<!fir.array<4x4xi32>>, !fir.shape<2>) -> !fir.array<4x4xi32>
! CHECK: %[[av:.*]] = fir.do_loop
! CHECK: fir.do_loop
! CHECK: = fir.array_fetch %[[a2]], %{{.*}}, %{{.*}} : (!fir.array<4x4xi32>, index, index) -> i32
! CHECK: = fir.array_fetch %[[a3]], %{{.*}}, %{{.*}} : (!fir.array<4x4xi32>, index, index) -> i32
! CHECK: = fir.array_update %{{.*}}, %{{.*}}, %{{.*}} : (!fir.array<4x4xi32>, i32, index, index) -> !fir.array<4x4xi32>
! CHECK : fir.array_merge_store %[[a1]], %[[av]] to %[[a]] : !fir.array<4x4xi32>, !fir.array<4x4xi32>, !fir.ref<!fir.array<4x4xi32>>
