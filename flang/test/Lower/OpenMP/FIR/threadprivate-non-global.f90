! This test checks lowering of OpenMP Threadprivate Directive.
! Test for non-character non-SAVEd non-initialized scalars with or without
! allocatable or pointer attribute in main program.

!RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -fopenmp %s -o - | FileCheck %s

program test
  integer :: x
  real :: y
  logical :: z
  complex :: w
  integer, pointer :: a
  real, allocatable :: b

!CHECK-DAG:  [[ADDR0:%.*]] = fir.address_of(@_QFEa) : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:  [[NEWADDR0:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.box<!fir.ptr<i32>>> -> !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:  [[ADDR1:%.*]] = fir.address_of(@_QFEb) : !fir.ref<!fir.box<!fir.heap<f32>>>
!CHECK-DAG:  [[NEWADDR1:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<!fir.box<!fir.heap<f32>>> -> !fir.ref<!fir.box<!fir.heap<f32>>>
!CHECK-DAG:  [[ADDR2:%.*]] = fir.address_of(@_QFEw) : !fir.ref<!fir.complex<4>>
!CHECK-DAG:  [[NEWADDR2:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<!fir.complex<4>> -> !fir.ref<!fir.complex<4>>
!CHECK-DAG:  [[ADDR3:%.*]] = fir.address_of(@_QFEx) : !fir.ref<i32>
!CHECK-DAG:  [[NEWADDR3:%.*]] = omp.threadprivate [[ADDR3]] : !fir.ref<i32> -> !fir.ref<i32>
!CHECK-DAG:  [[ADDR4:%.*]] = fir.address_of(@_QFEy) : !fir.ref<f32>
!CHECK-DAG:  [[NEWADDR4:%.*]] = omp.threadprivate [[ADDR4]] : !fir.ref<f32> -> !fir.ref<f32>
!CHECK-DAG:  [[ADDR5:%.*]] = fir.address_of(@_QFEz) : !fir.ref<!fir.logical<4>>
!CHECK-DAG:  [[NEWADDR5:%.*]] = omp.threadprivate [[ADDR5]] : !fir.ref<!fir.logical<4>> -> !fir.ref<!fir.logical<4>>
  !$omp threadprivate(x, y, z, w, a, b)

  call sub(a, b)

!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR3]] : !fir.ref<i32>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR4]] : !fir.ref<f32>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR5]] : !fir.ref<!fir.logical<4>>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR2]] : !fir.ref<!fir.complex<4>>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR1]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  print *, x, y, z, w, a, b

  !$omp parallel
!CHECK-DAG:    [[ADDR68:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<!fir.box<!fir.ptr<i32>>> -> !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:    [[ADDR69:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<!fir.box<!fir.heap<f32>>> -> !fir.ref<!fir.box<!fir.heap<f32>>>
!CHECK-DAG:    [[ADDR70:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<!fir.complex<4>> -> !fir.ref<!fir.complex<4>>
!CHECK-DAG:    [[ADDR71:%.*]] = omp.threadprivate [[ADDR3]] : !fir.ref<i32> -> !fir.ref<i32>
!CHECK-DAG:    [[ADDR72:%.*]] = omp.threadprivate [[ADDR4]] : !fir.ref<f32> -> !fir.ref<f32>
!CHECK-DAG:    [[ADDR73:%.*]] = omp.threadprivate [[ADDR5]] : !fir.ref<!fir.logical<4>> -> !fir.ref<!fir.logical<4>>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR71]] : !fir.ref<i32>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR72]] : !fir.ref<f32>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR73]] : !fir.ref<!fir.logical<4>>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR70]] : !fir.ref<!fir.complex<4>>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR68]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR69]] : !fir.ref<!fir.box<!fir.heap<f32>>>
    print *, x, y, z, w, a, b
  !$omp end parallel

!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR3]] : !fir.ref<i32>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR4]] : !fir.ref<f32>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR5]] : !fir.ref<!fir.logical<4>>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR2]] : !fir.ref<!fir.complex<4>>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR1]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  print *, x, y, z, w, a, b

!CHECK:  return

!CHECK-DAG: fir.global internal @_QFEa : !fir.box<!fir.ptr<i32>> {
!CHECK-DAG:   [[Z0:%.*]] = fir.zero_bits !fir.ptr<i32>
!CHECK-DAG:   [[E0:%.*]] = fir.embox [[Z0]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK-DAG:   fir.has_value [[E0]] : !fir.box<!fir.ptr<i32>>
!CHECK-DAG: }
!CHECK-DAG: fir.global internal @_QFEb : !fir.box<!fir.heap<f32>> {
!CHECK-DAG:   [[Z1:%.*]] = fir.zero_bits !fir.heap<f32>
!CHECK-DAG:   [[E1:%.*]] = fir.embox [[Z1]] : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
!CHECK-DAG:   fir.has_value [[E1]] : !fir.box<!fir.heap<f32>>
!CHECK-DAG: }
!CHECK-DAG: fir.global internal @_QFEw : !fir.complex<4> {
!CHECK-DAG:   [[Z2:%.*]] = fir.undefined !fir.complex<4>
!CHECK-DAG:   fir.has_value [[Z2]] : !fir.complex<4>
!CHECK-DAG: }
!CHECK-DAG: fir.global internal @_QFEx : i32 {
!CHECK-DAG:   [[Z3:%.*]] = fir.undefined i32
!CHECK-DAG:   fir.has_value [[Z3]] : i32
!CHECK-DAG: }
!CHECK-DAG: fir.global internal @_QFEy : f32 {
!CHECK-DAG:   [[Z4:%.*]] = fir.undefined f32
!CHECK-DAG:   fir.has_value [[Z4]] : f32
!CHECK-DAG: }
!CHECK-DAG: fir.global internal @_QFEz : !fir.logical<4> {
!CHECK-DAG:   [[Z5:%.*]] = fir.undefined !fir.logical<4>
!CHECK-DAG:   fir.has_value [[Z5]] : !fir.logical<4>
!CHECK-DAG: }
end
