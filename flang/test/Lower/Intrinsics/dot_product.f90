! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

! DOT_PROD
! CHECK-LABEL: dot_prod_int_default
! CHECK-SAME: %[[x:arg0]]: !fir.box<!fir.array<?xi32>>
! CHECK-SAME: %[[y:arg1]]: !fir.box<!fir.array<?xi32>>
! CHECK-SAME: %[[z:arg2]]: !fir.box<!fir.array<?xi32>>
subroutine dot_prod_int_default (x, y, z)
  integer, dimension(1:) :: x,y
  integer, dimension(1:) :: z
  ! CHECK-DAG: %[[x_conv:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[y_conv:.*]] = fir.convert %[[y]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[res:.*]] = fir.call @_FortranADotProductInteger4(%[[x_conv]], %[[y_conv]], %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_int_kind_1
! CHECK-SAME: %[[x:arg0]]: !fir.box<!fir.array<?xi8>>
! CHECK-SAME: %[[y:arg1]]: !fir.box<!fir.array<?xi8>>
! CHECK-SAME: %[[z:arg2]]: !fir.box<!fir.array<?xi8>>
subroutine dot_prod_int_kind_1 (x, y, z)
  integer(kind=1), dimension(1:) :: x,y
  integer(kind=1), dimension(1:) :: z
  ! CHECK-DAG: %[[x_conv:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xi8>>) -> !fir.box<none>
  ! CHECK-DAG: %[[y_conv:.*]] = fir.convert %[[y]] : (!fir.box<!fir.array<?xi8>>) -> !fir.box<none>
  ! CHECK-DAG: %[[res:.*]] = fir.call @_FortranADotProductInteger1(%[[x_conv]], %[[y_conv]], %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i8
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_int_kind_2
! CHECK-SAME: %[[x:arg0]]: !fir.box<!fir.array<?xi16>>
! CHECK-SAME: %[[y:arg1]]: !fir.box<!fir.array<?xi16>>
! CHECK-SAME: %[[z:arg2]]: !fir.box<!fir.array<?xi16>>
subroutine dot_prod_int_kind_2 (x, y, z)
  integer(kind=2), dimension(1:) :: x,y
  integer(kind=2), dimension(1:) :: z
  ! CHECK-DAG: %[[x_conv:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xi16>>) -> !fir.box<none>
  ! CHECK-DAG: %[[y_conv:.*]] = fir.convert %[[y]] : (!fir.box<!fir.array<?xi16>>) -> !fir.box<none>
  ! CHECK-DAG: %[[res:.*]] = fir.call @_FortranADotProductInteger2(%[[x_conv]], %[[y_conv]], %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i16
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_int_kind_4
! CHECK-SAME: %[[x:arg0]]: !fir.box<!fir.array<?xi32>>
! CHECK-SAME: %[[y:arg1]]: !fir.box<!fir.array<?xi32>>
! CHECK-SAME: %[[z:arg2]]: !fir.box<!fir.array<?xi32>>
subroutine dot_prod_int_kind_4 (x, y, z)
  integer(kind=4), dimension(1:) :: x,y
  integer(kind=4), dimension(1:) :: z
  ! CHECK-DAG: %[[x_conv:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[y_conv:.*]] = fir.convert %[[y]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[res:.*]] = fir.call @_FortranADotProductInteger4(%[[x_conv]], %[[y_conv]], %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_int_kind_8
! CHECK-SAME: %[[x:arg0]]: !fir.box<!fir.array<?xi64>>
! CHECK-SAME: %[[y:arg1]]: !fir.box<!fir.array<?xi64>>
! CHECK-SAME: %[[z:arg2]]: !fir.box<!fir.array<?xi64>>
subroutine dot_prod_int_kind_8 (x, y, z)
  integer(kind=8), dimension(1:) :: x,y
  integer(kind=8), dimension(1:) :: z
  ! CHECK-DAG: %[[x_conv:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xi64>>) -> !fir.box<none>
  ! CHECK-DAG: %[[y_conv:.*]] = fir.convert %[[y]] : (!fir.box<!fir.array<?xi64>>) -> !fir.box<none>
  ! CHECK-DAG: %[[res:.*]] = fir.call @_FortranADotProductInteger8(%[[x_conv]], %[[y_conv]], %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i64
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_int_kind_16
! CHECK-SAME: %[[x:arg0]]: !fir.box<!fir.array<?xi128>>
! CHECK-SAME: %[[y:arg1]]: !fir.box<!fir.array<?xi128>>
! CHECK-SAME: %[[z:arg2]]: !fir.box<!fir.array<?xi128>>
subroutine dot_prod_int_kind_16 (x, y, z)
  integer(kind=16), dimension(1:) :: x,y
  integer(kind=16), dimension(1:) :: z
  ! CHECK-DAG: %[[x_conv:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xi128>>) -> !fir.box<none>
  ! CHECK-DAG: %[[y_conv:.*]] = fir.convert %[[y]] : (!fir.box<!fir.array<?xi128>>) -> !fir.box<none>
  ! CHECK-DAG: %[[res:.*]] = fir.call @_FortranADotProductInteger16(%[[x_conv]], %[[y_conv]], %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i128
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_real_kind_default
! CHECK-SAME: %[[x:arg0]]: !fir.box<!fir.array<?xf32>>
! CHECK-SAME: %[[y:arg1]]: !fir.box<!fir.array<?xf32>>
! CHECK-SAME: %[[z:arg2]]: !fir.box<!fir.array<?xf32>>
subroutine dot_prod_real_kind_default (x, y, z)
  real, dimension(1:) :: x,y
  real, dimension(1:) :: z
  ! CHECK-DAG: %[[x_conv:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[y_conv:.*]] = fir.convert %[[y]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[res:.*]] = fir.call @_FortranADotProductReal4(%[[x_conv]], %[[y_conv]], %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f32
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_real_kind_4
! CHECK-SAME: %[[x:arg0]]: !fir.box<!fir.array<?xf32>>
! CHECK-SAME: %[[y:arg1]]: !fir.box<!fir.array<?xf32>>
! CHECK-SAME: %[[z:arg2]]: !fir.box<!fir.array<?xf32>>
subroutine dot_prod_real_kind_4 (x, y, z)
  real(kind=4), dimension(1:) :: x,y
  real(kind=4), dimension(1:) :: z
  ! CHECK-DAG: %[[x_conv:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[y_conv:.*]] = fir.convert %[[y]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[res:.*]] = fir.call @_FortranADotProductReal4(%[[x_conv]], %[[y_conv]], %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f32
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_real_kind_8
! CHECK-SAME: %[[x:arg0]]: !fir.box<!fir.array<?xf64>>
! CHECK-SAME: %[[y:arg1]]: !fir.box<!fir.array<?xf64>>
! CHECK-SAME: %[[z:arg2]]: !fir.box<!fir.array<?xf64>>
subroutine dot_prod_real_kind_8 (x, y, z)
  real(kind=8), dimension(1:) :: x,y
  real(kind=8), dimension(1:) :: z
  ! CHECK-DAG: %[[x_conv:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xf64>>) -> !fir.box<none>
  ! CHECK-DAG: %[[y_conv:.*]] = fir.convert %[[y]] : (!fir.box<!fir.array<?xf64>>) -> !fir.box<none>
  ! CHECK-DAG: %[[res:.*]] = fir.call @_FortranADotProductReal8(%[[x_conv]], %[[y_conv]], %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f64
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_real_kind_10
! CHECK-SAME: %[[x:arg0]]: !fir.box<!fir.array<?xf80>>
! CHECK-SAME: %[[y:arg1]]: !fir.box<!fir.array<?xf80>>
! CHECK-SAME: %[[z:arg2]]: !fir.box<!fir.array<?xf80>>
subroutine dot_prod_real_kind_10 (x, y, z)
  real(kind=10), dimension(1:) :: x,y
  real(kind=10), dimension(1:) :: z
  ! CHECK-DAG: %[[x_conv:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xf80>>) -> !fir.box<none>
  ! CHECK-DAG: %[[y_conv:.*]] = fir.convert %[[y]] : (!fir.box<!fir.array<?xf80>>) -> !fir.box<none>
  ! CHECK-DAG: %[[res:.*]] = fir.call @_FortranADotProductReal10(%[[x_conv]], %[[y_conv]], %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f80
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_real_kind_16
! CHECK-SAME: %[[x:arg0]]: !fir.box<!fir.array<?xf128>>
! CHECK-SAME: %[[y:arg1]]: !fir.box<!fir.array<?xf128>>
! CHECK-SAME: %[[z:arg2]]: !fir.box<!fir.array<?xf128>>
subroutine dot_prod_real_kind_16 (x, y, z)
  real(kind=16), dimension(1:) :: x,y
  real(kind=16), dimension(1:) :: z
  ! CHECK-DAG: %[[x_conv:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xf128>>) -> !fir.box<none>
  ! CHECK-DAG: %[[y_conv:.*]] = fir.convert %[[y]] : (!fir.box<!fir.array<?xf128>>) -> !fir.box<none>
  ! CHECK-DAG: %[[res:.*]] = fir.call @_FortranADotProductReal16(%[[x_conv]], %[[y_conv]], %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f128
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_double_default
! CHECK-SAME: %[[x:arg0]]: !fir.box<!fir.array<?xf64>>
! CHECK-SAME: %[[y:arg1]]: !fir.box<!fir.array<?xf64>>
! CHECK-SAME: %[[z:arg2]]: !fir.box<!fir.array<?xf64>>
subroutine dot_prod_double_default (x, y, z)
  double precision, dimension(1:) :: x,y
  double precision, dimension(1:) :: z
  ! CHECK-DAG: %[[x_conv:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xf64>>) -> !fir.box<none>
  ! CHECK-DAG: %[[y_conv:.*]] = fir.convert %[[y]] : (!fir.box<!fir.array<?xf64>>) -> !fir.box<none>
  ! CHECK-DAG: %[[res:.*]] = fir.call @_FortranADotProductReal8(%[[x_conv]], %[[y_conv]], %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f64
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_complex_default
! CHECK-SAME: %[[x:arg0]]: !fir.box<!fir.array<?xcomplex<f32>>>
! CHECK-SAME: %[[y:arg1]]: !fir.box<!fir.array<?xcomplex<f32>>>
! CHECK-SAME: %[[z:arg2]]: !fir.box<!fir.array<?xcomplex<f32>>>
subroutine dot_prod_complex_default (x, y, z)
  complex, dimension(1:) :: x,y
  complex, dimension(1:) :: z
  ! CHECK-DAG: %[[res:.*]] = fir.alloca complex<f32>
  ! CHECK-DAG: %[[x_conv:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xcomplex<f32>>>) -> !fir.box<none>
  ! CHECK-DAG: %[[y_conv:.*]] = fir.convert %[[y]] : (!fir.box<!fir.array<?xcomplex<f32>>>) -> !fir.box<none>
  ! CHECK-DAG: fir.call @_FortranACppDotProductComplex4(%[[res]], %[[x_conv]], %[[y_conv]], %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.ref<complex<f32>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_complex_kind_4
! CHECK-SAME: %[[x:arg0]]: !fir.box<!fir.array<?xcomplex<f32>>>
! CHECK-SAME: %[[y:arg1]]: !fir.box<!fir.array<?xcomplex<f32>>>
! CHECK-SAME: %[[z:arg2]]: !fir.box<!fir.array<?xcomplex<f32>>>
subroutine dot_prod_complex_kind_4 (x, y, z)
  complex(kind=4), dimension(1:) :: x,y
  complex(kind=4), dimension(1:) :: z
  ! CHECK-DAG: %[[res:.*]] = fir.alloca complex<f32>
  ! CHECK-DAG: %[[x_conv:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xcomplex<f32>>>) -> !fir.box<none>
  ! CHECK-DAG: %[[y_conv:.*]] = fir.convert %[[y]] : (!fir.box<!fir.array<?xcomplex<f32>>>) -> !fir.box<none>
  ! CHECK-DAG: fir.call @_FortranACppDotProductComplex4(%[[res]], %[[x_conv]], %[[y_conv]], %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.ref<complex<f32>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_complex_kind_8
! CHECK-SAME: %[[x:arg0]]: !fir.box<!fir.array<?xcomplex<f64>>>
! CHECK-SAME: %[[y:arg1]]: !fir.box<!fir.array<?xcomplex<f64>>>
! CHECK-SAME: %[[z:arg2]]: !fir.box<!fir.array<?xcomplex<f64>>>
subroutine dot_prod_complex_kind_8 (x, y, z)
  complex(kind=8), dimension(1:) :: x,y
  complex(kind=8), dimension(1:) :: z
  ! CHECK-DAG: %[[res:.*]] = fir.alloca complex<f64>
  ! CHECK-DAG: %[[x_conv:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xcomplex<f64>>>) -> !fir.box<none>
  ! CHECK-DAG: %[[y_conv:.*]] = fir.convert %[[y]] : (!fir.box<!fir.array<?xcomplex<f64>>>) -> !fir.box<none>
  ! CHECK-DAG: fir.call @_FortranACppDotProductComplex8(%[[res]], %[[x_conv]], %[[y_conv]], %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.ref<complex<f64>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_complex_kind_10
! CHECK-SAME: %[[x:arg0]]: !fir.box<!fir.array<?xcomplex<f80>>>
! CHECK-SAME: %[[y:arg1]]: !fir.box<!fir.array<?xcomplex<f80>>>
! CHECK-SAME: %[[z:arg2]]: !fir.box<!fir.array<?xcomplex<f80>>>
subroutine dot_prod_complex_kind_10 (x, y, z)
  complex(kind=10), dimension(1:) :: x,y
  complex(kind=10), dimension(1:) :: z
  ! CHECK-DAG: %[[res:.*]] = fir.alloca complex<f80>
  ! CHECK-DAG: %[[x_conv:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xcomplex<f80>>>) -> !fir.box<none>
  ! CHECK-DAG: %[[y_conv:.*]] = fir.convert %[[y]] : (!fir.box<!fir.array<?xcomplex<f80>>>) -> !fir.box<none>
  ! CHECK-DAG: fir.call @_FortranACppDotProductComplex10(%[[res]], %[[x_conv]], %[[y_conv]], %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.ref<complex<f80>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_complex_kind_16
! CHECK-SAME: %[[x:arg0]]: !fir.box<!fir.array<?xcomplex<f128>>>
! CHECK-SAME: %[[y:arg1]]: !fir.box<!fir.array<?xcomplex<f128>>>
! CHECK-SAME: %[[z:arg2]]: !fir.box<!fir.array<?xcomplex<f128>>>
subroutine dot_prod_complex_kind_16 (x, y, z)
  complex(kind=16), dimension(1:) :: x,y
  complex(kind=16), dimension(1:) :: z
  ! CHECK-DAG: %[[res:.*]] = fir.alloca complex<f128>
  ! CHECK-DAG: %[[x_conv:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xcomplex<f128>>>) -> !fir.box<none>
  ! CHECK-DAG: %[[y_conv:.*]] = fir.convert %[[y]] : (!fir.box<!fir.array<?xcomplex<f128>>>) -> !fir.box<none>
  ! CHECK-DAG: fir.call @_FortranACppDotProductComplex16(%[[res]], %[[x_conv]], %[[y_conv]], %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.ref<complex<f128>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_logical
! CHECK-SAME: %[[x:arg0]]: !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK-SAME: %[[y:arg1]]: !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK-SAME: %[[z:arg2]]: !fir.box<!fir.array<?x!fir.logical<4>>>
subroutine dot_prod_logical (x, y, z)
  logical, dimension(1:) :: x,y
  logical, dimension(1:) :: z
  ! CHECK-DAG: %[[x_conv:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
  ! CHECK-DAG: %[[y_conv:.*]] = fir.convert %[[y]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
  ! CHECK-DAG: %[[res:.*]] = fir.call @_FortranADotProductLogical(%[[x_conv]], %[[y_conv]], %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i1
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_product_mixed_int_real
! CHECK-SAME: %[[x:arg0]]: !fir.box<!fir.array<?xi32>>
! CHECK-SAME: %[[y:arg1]]: !fir.box<!fir.array<?xf32>>
! CHECK-SAME: %[[z:arg2]]: !fir.box<!fir.array<?xf32>>
subroutine dot_product_mixed_int_real(x, y, z)
  integer, dimension(1:) :: x
  real, dimension(1:) :: y, z
  ! CHECK-DAG: %[[x_conv:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[y_conv:.*]] = fir.convert %[[y]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[res:.*]] = fir.call @_FortranADotProductReal4(%[[x_conv]], %[[y_conv]], %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f32
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_product_mixed_int_complex
! CHECK-SAME: %[[x:arg0]]: !fir.box<!fir.array<?xi32>>
! CHECK-SAME: %[[y:arg1]]: !fir.box<!fir.array<?xcomplex<f32>>>
! CHECK-SAME: %[[z:arg2]]: !fir.box<!fir.array<?xcomplex<f32>>>
subroutine dot_product_mixed_int_complex(x, y, z)
  integer, dimension(1:) :: x
  complex, dimension(1:) :: y, z
  ! CHECK-DAG: %[[res:.*]] = fir.alloca complex<f32>
  ! CHECK-DAG: %[[x_conv:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[y_conv:.*]] = fir.convert %[[y]] : (!fir.box<!fir.array<?xcomplex<f32>>>) -> !fir.box<none>
  ! CHECK-DAG: fir.call @_FortranACppDotProductComplex4(%[[res]], %[[x_conv]], %[[y_conv]], %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.ref<complex<f32>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_product_mixed_real_complex
! CHECK-SAME: %[[x:arg0]]: !fir.box<!fir.array<?xf32>>
! CHECK-SAME: %[[y:arg1]]: !fir.box<!fir.array<?xcomplex<f32>>>
! CHECK-SAME: %[[z:arg2]]: !fir.box<!fir.array<?xcomplex<f32>>>
subroutine dot_product_mixed_real_complex(x, y, z)
  real, dimension(1:) :: x
  complex, dimension(1:) :: y, z
  ! CHECK-DAG: %[[res:.*]] = fir.alloca complex<f32>
  ! CHECK-DAG: %[[x_conv:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[y_conv:.*]] = fir.convert %[[y]] : (!fir.box<!fir.array<?xcomplex<f32>>>) -> !fir.box<none>
  ! CHECK-DAG: fir.call @_FortranACppDotProductComplex4(%[[res]], %[[x_conv]], %[[y_conv]], %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.ref<complex<f32>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  z = dot_product(x,y)
end subroutine
