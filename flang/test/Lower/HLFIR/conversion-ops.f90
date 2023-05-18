! Test lowering of intrinsic conversions to HLFIR
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s

subroutine test
  integer(4) :: i4
  integer(8) :: i8
  real(4) :: r4
  real(8) :: r8
  complex(4) :: z4
  complex(8) :: z8

  logical(4) :: l4
  logical(8) :: l8

  i4 = i8
! CHECK:  fir.convert %{{.*}} : (i64) -> i32
  i4 = r4
! CHECK:  fir.convert %{{.*}} : (f32) -> i32
  i4 = r8
! CHECK:  fir.convert %{{.*}} : (f64) -> i32
  i4 = z4
! CHECK:  %[[VAL_23:.*]] = fir.extract_value %{{.*}}, [0 : index] : (!fir.complex<4>) -> f32
! CHECK:  fir.convert %[[VAL_23]] : (f32) -> i32
  i4 = z8
! CHECK:  %[[VAL_26:.*]] = fir.extract_value %{{.*}}, [0 : index] : (!fir.complex<8>) -> f64
! CHECK:  fir.convert %[[VAL_26]] : (f64) -> i32

  r4 = i4
! CHECK:  fir.convert %{{.*}} : (i32) -> f32
  r4 = i8
! CHECK:  fir.convert %{{.*}} : (i64) -> f32
  r4 = r8
! CHECK:  fir.convert %{{.*}} : (f64) -> f32
  r4 = z4
! CHECK:  fir.extract_value %{{.*}}, [0 : index] : (!fir.complex<4>) -> f32
  r4 = z8
! CHECK:  %[[VAL_36:.*]] = fir.load %{{.*}} : !fir.ref<!fir.complex<8>>
! CHECK:  %[[VAL_37:.*]] = fir.extract_value %[[VAL_36]], [0 : index] : (!fir.complex<8>) -> f64
! CHECK:  fir.convert %[[VAL_37]] : (f64) -> f32

  z4 = i4
! CHECK:  %[[VAL_40:.*]] = fir.convert %{{.*}} : (i32) -> f32
! CHECK:  %[[VAL_41:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:  %[[VAL_42:.*]] = fir.undefined !fir.complex<4>
! CHECK:  %[[VAL_43:.*]] = fir.insert_value %[[VAL_42]], %[[VAL_40]], [0 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
! CHECK:  fir.insert_value %[[VAL_43]], %[[VAL_41]], [1 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
  z4 = i8
! CHECK:  %[[VAL_46:.*]] = fir.convert %{{.*}} : (i64) -> f32
! CHECK:  %[[VAL_47:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:  %[[VAL_48:.*]] = fir.undefined !fir.complex<4>
! CHECK:  %[[VAL_49:.*]] = fir.insert_value %[[VAL_48]], %[[VAL_46]], [0 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
! CHECK:  fir.insert_value %[[VAL_49]], %[[VAL_47]], [1 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
  z4 = r4
! CHECK:  %[[VAL_52:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:  %[[VAL_53:.*]] = fir.undefined !fir.complex<4>
! CHECK:  %[[VAL_54:.*]] = fir.insert_value %[[VAL_53]], %{{.*}}, [0 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
! CHECK:  fir.insert_value %[[VAL_54]], %[[VAL_52]], [1 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
  z4 = r8
! CHECK:  %[[VAL_57:.*]] = fir.convert %{{.*}} : (f64) -> f32
! CHECK:  %[[VAL_58:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:  %[[VAL_59:.*]] = fir.undefined !fir.complex<4>
! CHECK:  %[[VAL_60:.*]] = fir.insert_value %[[VAL_59]], %[[VAL_57]], [0 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
! CHECK:  fir.insert_value %[[VAL_60]], %[[VAL_58]], [1 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
  z4 = z8
! CHECK:  fir.convert %{{.*}} : (!fir.complex<8>) -> !fir.complex<4>

  l4 = l8
! CHECK:  fir.convert %{{.*}} : (!fir.logical<8>) -> !fir.logical<4>
end subroutine
