! Test lowering of references to pointers
! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Assigning/reading to scalar pointer target.
! CHECK-LABEL: func @_QPscal_ptr(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.ptr<f32>>>)
subroutine scal_ptr(p)
  real, pointer :: p
  real :: x
  ! CHECK: %[[boxload:.*]] = fir.load %[[arg0]]
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[boxload]]
  ! CHECK: fir.store %{{.*}} to %[[addr]]
  p = 3.

  ! CHECK: %[[boxload2:.*]] = fir.load %[[arg0]]
  ! CHECK: %[[addr2:.*]] = fir.box_addr %[[boxload2]]
  ! CHECK: %[[val:.*]] = fir.load %[[addr2]]
  ! CHECK: fir.store %[[val]] to %{{.*}}
  x = p
end subroutine

! Assigning/reading scalar character pointer target.
! CHECK-LABEL: func @_QPchar_ptr(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,12>>>>)
subroutine char_ptr(p)
  character(12), pointer :: p
  character(12) :: x

  ! CHECK: %[[boxload:.*]] = fir.load %[[arg0]]
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[boxload]]
  ! CHECK-DAG: %[[str:.*]] = fir.address_of(@_QQcl.68656C6C6F20776F726C6421) : !fir.ref<!fir.char<1,12>>
  ! CHECK-DAG: %[[one:.*]] = constant 1
  ! CHECK-DAG: %[[size:.*]] = fir.convert %{{.*}} : (index) -> i64
  ! CHECK: %[[count:.*]] = muli %[[one]], %[[size]] : i64
  ! CHECK: %[[dst:.*]] = fir.convert %[[addr]] : (!fir.ptr<!fir.char<1,12>>) -> !fir.ref<i8>
  ! CHECK: %[[src:.*]] = fir.convert %[[str]] : (!fir.ref<!fir.char<1,12>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0i8.p0i8.i64(%[[dst]], %[[src]], %5, %false) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  p = "hello world!"

  ! CHECK: %[[boxload2:.*]] = fir.load %[[arg0]]
  ! CHECK: %[[addr2:.*]] = fir.box_addr %[[boxload2]]
  ! CHECK: %[[count:.*]] = muli %{{.*}}, %{{.*}} : i64
  ! CHECK: %[[dst:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.char<1,12>>) -> !fir.ref<i8>
  ! CHECK: %[[src:.*]] = fir.convert %[[addr2]] : (!fir.ptr<!fir.char<1,12>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0i8.p0i8.i64(%[[dst]], %[[src]], %[[count]], %{{.*}}) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  x = p
end subroutine

! Reading from pointer in array expression
! CHECK-LABEL: func @_QParr_ptr_read(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
subroutine arr_ptr_read(p)
  real, pointer :: p(:)
  real :: x(100)
  ! CHECK: %[[boxload:.*]] = fir.load %[[arg0]]
  ! CHECK: %[[dims:.*]]:3 = fir.box_dims %[[boxload]], %c0{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
  ! CHECK: %[[lb:.*]] = fir.shift %[[dims]]#0 : (index) -> !fir.shift<1>
  ! CHECK: fir.array_load %[[boxload]](%[[lb]]) : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>) -> !fir.array<?xf32>
  x = p
end subroutine

! Reading from contiguous pointer in array expression
! CHECK-LABEL: func @_QParr_contig_ptr_read(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.contiguous})
subroutine arr_contig_ptr_read(p)
  real, pointer, contiguous :: p(:)
  real :: x(100)
  ! CHECK: %[[boxload:.*]] = fir.load %[[arg0]]
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[boxload]], %c0{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[addr:.*]] = fir.box_addr %[[boxload]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
  ! CHECK-DAG: %[[shape:.*]] = fir.shape_shift %[[dims]]#0, %[[dims]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK: fir.array_load %[[addr]](%[[shape]]) : (!fir.ptr<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.array<?xf32>
  x = p
end subroutine

! Assigning to pointer target in array expression
! CHECK-LABEL: func @_QParr_ptr_target_write(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
subroutine arr_ptr_target_write(p)
  real, pointer :: p(:)
  real :: x(100)
  ! CHECK-DAG: %[[boxload:.*]] = fir.load %[[arg0]]
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[boxload]], %c0{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[lb:.*]] = fir.shift %[[dims]]#0 : (index) -> !fir.shift<1>
  ! CHECK-DAG: %[[slice:.*]] = fir.slice %c2{{.*}}, %c601{{.*}}, %c6{{.*}}
  ! CHECK: %[[arrayld:.*]] = fir.array_load %[[boxload]](%[[lb]]) [%[[slice]]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>, !fir.slice<1>) -> !fir.array<?xf32>
  p(2:601:6) = x
  ! CHECK: fir.array_merge_store %[[arrayld]], %{{.*}} to %[[boxload]] : !fir.box<!fir.ptr<!fir.array<?xf32>>>
end subroutine

! Assigning to contiguous pointer target in array expression
! CHECK-LABEL: func @_QParr_contig_ptr_target_write(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.contiguous})
subroutine arr_contig_ptr_target_write(p)
  real, pointer, contiguous :: p(:)
  real :: x(100)
  ! CHECK: %[[boxload:.*]] = fir.load %[[arg0]]
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[boxload]], %c0{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[addr:.*]] = fir.box_addr %[[boxload]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
  ! CHECK-DAG: %[[shape:.*]] = fir.shape_shift %[[dims]]#0, %[[dims]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK-DAG: %[[slice:.*]] = fir.slice %c2{{.*}}, %c601{{.*}}, %c6{{.*}}
  ! CHECK: %[[arrayld:.*]] = fir.array_load %[[addr]](%[[shape]]) [%[[slice]]] : (!fir.ptr<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.array<?xf32>
  p(2:601:6) = x
  ! CHECK: fir.array_merge_store %[[arrayld]], %{{.*}} to %[[addr]] : !fir.ptr<!fir.array<?xf32>>
end subroutine
