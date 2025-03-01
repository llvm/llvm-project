! RUN: %flang_fc1 -emit-fir -O0 %s -o - | FileCheck %s --check-prefixes=CHECK%if target=x86_64{{.*}} %{,CHECK-KIND10%}%if flang-supports-f128-math %{,CHECK-KIND16%}

! DOT_PROD
! CHECK-LABEL: dot_prod_int_default
subroutine dot_prod_int_default (x, y, z)
  integer, dimension(1:) :: x,y
  integer, dimension(1:) :: z
  ! CHECK: %[[x1:.*]] = fir.declare{{.*}}x"
  ! CHECK: %[[x:.*]] = fir.rebox %[[x1]]{{.*}}
  ! CHECK: %[[y1:.*]] = fir.declare{{.*}}y"
  ! CHECK: %[[y:.*]] = fir.rebox %[[y1]]{{.*}}
  ! CHECK: %[[z1:.*]] = fir.declare{{.*}}z"
  ! CHECK: %[[z:.*]] = fir.rebox %[[z1]]{{.*}}
  ! CHECK-DAG: %[[x_conv:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[y_conv:.*]] = fir.convert %[[y]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
  ! CHECK-DAG: %[[res:.*]] = fir.call @_FortranADotProductInteger4(%[[x_conv]], %[[y_conv]], %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_int_kind_1
subroutine dot_prod_int_kind_1 (x, y, z)
  integer(kind=1), dimension(1:) :: x,y
  integer(kind=1), dimension(1:) :: z
  ! CHECK: fir.call @_FortranADotProductInteger1(%{{.*}}, %{{.*}}, %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i8
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_int_kind_2
subroutine dot_prod_int_kind_2 (x, y, z)
  integer(kind=2), dimension(1:) :: x,y
  integer(kind=2), dimension(1:) :: z
  ! CHECK: fir.call @_FortranADotProductInteger2(%{{.*}}, %{{.*}}, %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i16
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_int_kind_4
subroutine dot_prod_int_kind_4 (x, y, z)
  integer(kind=4), dimension(1:) :: x,y
  integer(kind=4), dimension(1:) :: z
  ! CHECK: fir.call @_FortranADotProductInteger4(%{{.*}}, %{{.*}}, %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_int_kind_8
subroutine dot_prod_int_kind_8 (x, y, z)
  integer(kind=8), dimension(1:) :: x,y
  integer(kind=8), dimension(1:) :: z
  ! CHECK: fir.call @_FortranADotProductInteger8(%{{.*}}, %{{.*}}, %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i64
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_int_kind_16
subroutine dot_prod_int_kind_16 (x, y, z)
  integer(kind=16), dimension(1:) :: x,y
  integer(kind=16), dimension(1:) :: z
  ! CHECK: fir.call @_FortranADotProductInteger16(%{{.*}}, %{{.*}}, %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i128
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_real_kind_default
subroutine dot_prod_real_kind_default (x, y, z)
  real, dimension(1:) :: x,y
  real, dimension(1:) :: z
  ! CHECK: fir.call @_FortranADotProductReal4(%{{.*}}, %{{.*}}, %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f32
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_real_kind_4
subroutine dot_prod_real_kind_4 (x, y, z)
  real(kind=4), dimension(1:) :: x,y
  real(kind=4), dimension(1:) :: z
  ! CHECK: fir.call @_FortranADotProductReal4(%{{.*}}, %{{.*}}, %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f32
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_real_kind_8
subroutine dot_prod_real_kind_8 (x, y, z)
  real(kind=8), dimension(1:) :: x,y
  real(kind=8), dimension(1:) :: z
  ! CHECK: fir.call @_FortranADotProductReal8(%{{.*}}, %{{.*}}, %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f64
  z = dot_product(x,y)
end subroutine

! CHECK-KIND10-LABEL: dot_prod_real_kind_10
subroutine dot_prod_real_kind_10 (x, y, z)
  integer, parameter :: kind10 = merge(10, 4, selected_real_kind(p=18).eq.10)
  real(kind=kind10), dimension(1:) :: x,y
  real(kind=kind10), dimension(1:) :: z
  ! CHECK-KIND10: fir.call @_FortranADotProductReal10(%{{.*}}, %{{.*}}, %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f80
  z = dot_product(x,y)
end subroutine

! CHECK-KIND16-LABEL: dot_prod_real_kind_16
subroutine dot_prod_real_kind_16 (x, y, z)
  integer, parameter :: kind16 = merge(16, 4, selected_real_kind(p=33).eq.16)
  real(kind=kind16), dimension(1:) :: x,y
  real(kind=kind16), dimension(1:) :: z
  ! CHECK-KIND16: fir.call @_FortranADotProductReal16(%{{.*}}, %{{.*}}, %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f128
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_double_default
subroutine dot_prod_double_default (x, y, z)
  double precision, dimension(1:) :: x,y
  double precision, dimension(1:) :: z
  ! CHECK: fir.call @_FortranADotProductReal8(%{{.*}}, %{{.*}}, %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f64
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_complex_default
subroutine dot_prod_complex_default (x, y, z)
  complex, dimension(1:) :: x,y
  complex, dimension(1:) :: z
  ! CHECK: %[[res:.*]] = fir.alloca complex<f32>
  ! CHECK: fir.call @_FortranACppDotProductComplex4(%[[res]], %{{.*}}, %{{.*}}, %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.ref<complex<f32>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_complex_kind_4
subroutine dot_prod_complex_kind_4 (x, y, z)
  complex(kind=4), dimension(1:) :: x,y
  complex(kind=4), dimension(1:) :: z
  ! CHECK: %[[res:.*]] = fir.alloca complex<f32>
  ! CHECK: fir.call @_FortranACppDotProductComplex4(%[[res]], %{{.*}}, %{{.*}}, %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.ref<complex<f32>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_complex_kind_8
subroutine dot_prod_complex_kind_8 (x, y, z)
  complex(kind=8), dimension(1:) :: x,y
  complex(kind=8), dimension(1:) :: z
  ! CHECK: %[[res:.*]] = fir.alloca complex<f64>
  ! CHECK: fir.call @_FortranACppDotProductComplex8(%[[res]], %{{.*}}, %{{.*}}, %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.ref<complex<f64>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
  z = dot_product(x,y)
end subroutine

! CHECK-KIND10-LABEL: dot_prod_complex_kind_10
subroutine dot_prod_complex_kind_10 (x, y, z)
  integer, parameter :: kind10 = merge(10, 4, selected_real_kind(p=18).eq.10)
  complex(kind=kind10), dimension(1:) :: x,y
  complex(kind=kind10), dimension(1:) :: z
  ! CHECK-KIND10: %[[res:.*]] = fir.alloca complex<f80>
  ! CHECK-KIND10: fir.call @_FortranACppDotProductComplex10(%[[res]], %{{.*}}, %{{.*}}, %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.ref<complex<f80>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
  z = dot_product(x,y)
end subroutine

! CHECK-KIND16-LABEL: dot_prod_complex_kind_16
subroutine dot_prod_complex_kind_16 (x, y, z)
  integer, parameter :: kind16 = merge(16, 4, selected_real_kind(p=33).eq.16)
  complex(kind=kind16), dimension(1:) :: x,y
  complex(kind=kind16), dimension(1:) :: z
  ! CHECK-KIND16: %[[res:.*]] = fir.alloca complex<f128>
  ! CHECK-KIND16: fir.call @_FortranACppDotProductComplex16(%[[res]], %{{.*}}, %{{.*}}, %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.ref<complex<f128>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_prod_logical
subroutine dot_prod_logical (x, y, z)
  logical, dimension(1:) :: x,y
  logical, dimension(1:) :: z
  ! CHECK: fir.call @_FortranADotProductLogical(%{{.*}}, %{{.*}}, %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i1
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_product_mixed_int_real
subroutine dot_product_mixed_int_real(x, y, z)
  integer, dimension(1:) :: x
  real, dimension(1:) :: y, z
  ! CHECK: fir.call @_FortranADotProductReal4(%{{.*}}, %{{.*}}, %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> f32
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_product_mixed_int_complex
subroutine dot_product_mixed_int_complex(x, y, z)
  integer, dimension(1:) :: x
  complex, dimension(1:) :: y, z
  ! CHECK: %[[res:.*]] = fir.alloca complex<f32>
  ! CHECK: fir.call @_FortranACppDotProductComplex4(%[[res]], %{{.*}}, %{{.*}}, %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.ref<complex<f32>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
  z = dot_product(x,y)
end subroutine

! CHECK-LABEL: dot_product_mixed_real_complex
subroutine dot_product_mixed_real_complex(x, y, z)
  real, dimension(1:) :: x
  complex, dimension(1:) :: y, z
  ! CHECK: %[[res:.*]] = fir.alloca complex<f32>
  ! CHECK: fir.call @_FortranACppDotProductComplex4(%[[res]], %{{.*}}, %{{.*}}, %{{[0-9]+}}, %{{.*}}) {{.*}}: (!fir.ref<complex<f32>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
  z = dot_product(x,y)
end subroutine
