! Note: flang will issue warnings for the following subroutines. These
! are accepted regardless to maintain backwards compatibility with
! other Fortran implementations.

! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPs1() {
! CHECK: %[[cast:.*]] = fir.convert %{{.*}} : (!fir.ref<f32>) -> !fir.ref<!fir.char<1,?>>
! CHECK: %[[undef:.*]] = fir.undefined index
! CHECK: %[[box:.*]] = fir.emboxchar %[[cast]], %[[undef]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK: fir.call @_QPs3(%[[box]]) {{.*}}: (!fir.boxchar<1>) -> ()

! Pass a REAL by reference to a subroutine expecting a CHARACTER
subroutine s1
  call s3(r)
end subroutine s1

! CHECK-LABEL: func @_QPs2(
! CHECK: %[[ptr:.*]] = fir.box_addr %{{.*}} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
! CHECK: %[[cast:.*]] = fir.convert %[[ptr]] : (!fir.ptr<f32>) -> !fir.ref<!fir.char<1,?>>
! CHECK: %[[undef:.*]] = fir.undefined index
! CHECK: %[[box:.*]] = fir.emboxchar %[[cast]], %[[undef]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK: fir.call @_QPs3(%[[box]]) {{.*}}: (!fir.boxchar<1>) -> ()

! Pass a REAL, POINTER data reference to a subroutine expecting a CHARACTER
subroutine s2(p)
  real, pointer :: p
  call s3(p)
end subroutine s2

! CHECK-LABEL: func @_QPs3(
! CHECK-SAME: !fir.boxchar<1>
subroutine s3(c)
  character(8) c
end subroutine s3
