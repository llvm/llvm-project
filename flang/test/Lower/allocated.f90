! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: allocated_test
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.heap<f32>>>{{.*}}, %[[arg1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>{{.*}})
subroutine allocated_test(scalar, array)
    real, allocatable  :: scalar, array(:)
    ! CHECK: %[[scalar:.*]] = fir.load %[[arg0]] : !fir.ref<!fir.box<!fir.heap<f32>>>
    ! CHECK: %[[addr0:.*]] = fir.box_addr %[[scalar]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
    ! CHECK: %[[addrToInt0:.*]] = fir.convert %[[addr0]]
    ! CHECK: cmpi ne, %[[addrToInt0]], %c0{{.*}}
    print *, allocated(scalar)
    ! CHECK: %[[array:.*]] = fir.load %[[arg1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    ! CHECK: %[[addr1:.*]] = fir.box_addr %[[array]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
    ! CHECK: %[[addrToInt1:.*]] = fir.convert %[[addr1]]
    ! CHECK: cmpi ne, %[[addrToInt1]], %c0{{.*}}
    print *, allocated(array)
  end subroutine
  