! RUN: %flang_fc1 -emit-hlfir %s -o - -mllvm -math-runtime=fast | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir %s -o - -mllvm -math-runtime=precise | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_real4
! CHECK-SAME: (%[[argx:.*]]: !fir.ref<f32>{{.*}}, %[[argn:.*]]: !fir.ref<i32>{{.*}}) -> f32
function test_real4(x, n)
  real :: x, test_real4
  integer :: n

  ! CHECK: %[[n_decl:.*]]:2 = hlfir.declare %[[argn]] {{.*}}
  ! CHECK: %[[res_decl:.*]]:2 = hlfir.declare %{{.*}} {{.*}}
  ! CHECK: %[[x_decl:.*]]:2 = hlfir.declare %[[argx]] {{.*}}
  ! CHECK-DAG: %[[n:.*]] = fir.load %[[n_decl]]#0 : !fir.ref<i32>
  ! CHECK-DAG: %[[x:.*]] = fir.load %[[x_decl]]#0 : !fir.ref<f32>
  ! CHECK: %[[jn:.*]] = fir.call @jnf(%[[n]], %[[x]]) {{.*}} : (i32, f32) -> f32
  ! CHECK: hlfir.assign %[[jn]] to %[[res_decl]]#0 : f32, !fir.ref<f32>
  test_real4 = bessel_jn(n, x)
end function

! CHECK-LABEL: func.func @_QPtest_real8
! CHECK-SAME: (%[[argx:.*]]: !fir.ref<f64>{{.*}}, %[[argn:.*]]: !fir.ref<i32>{{.*}}) -> f64
function test_real8(x, n)
  real(8) :: x, test_real8
  integer :: n

  ! CHECK: %[[n_decl:.*]]:2 = hlfir.declare %[[argn]] {{.*}}
  ! CHECK: %[[res_decl:.*]]:2 = hlfir.declare %{{.*}} {{.*}}
  ! CHECK: %[[x_decl:.*]]:2 = hlfir.declare %[[argx]] {{.*}}
  ! CHECK-DAG: %[[n:.*]] = fir.load %[[n_decl]]#0 : !fir.ref<i32>
  ! CHECK-DAG: %[[x:.*]] = fir.load %[[x_decl]]#0 : !fir.ref<f64>
  ! CHECK: %[[jn:.*]] = fir.call @jn(%[[n]], %[[x]]) {{.*}} : (i32, f64) -> f64
  ! CHECK: hlfir.assign %[[jn]] to %[[res_decl]]#0 : f64, !fir.ref<f64>
  test_real8 = bessel_jn(n, x)
end function

! CHECK-LABEL: func.func @_QPtest_transformational_real4
! CHECK-SAME: %[[argx:.*]]: !fir.ref<f32>{{.*}}, %[[argn1:.*]]: !fir.ref<i32>{{.*}}, %[[argn2:.*]]: !fir.ref<i32>{{.*}}, %[[argr:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}
subroutine test_transformational_real4(x, n1, n2, r)
  real(4) :: x
  integer :: n1, n2
  real(4) :: r(:)

  ! CHECK: %[[n1_decl:.*]]:2 = hlfir.declare %[[argn1]] {{.*}}
  ! CHECK: %[[n2_decl:.*]]:2 = hlfir.declare %[[argn2]] {{.*}}
  ! CHECK: %[[r_decl:.*]]:2 = hlfir.declare %[[argr]] {{.*}}
  ! CHECK: %[[x_decl:.*]]:2 = hlfir.declare %[[argx]] {{.*}}
  ! CHECK-DAG: %[[n1:.*]] = fir.load %[[n1_decl]]#0 : !fir.ref<i32>
  ! CHECK-DAG: %[[n2:.*]] = fir.load %[[n2_decl]]#0 : !fir.ref<i32>
  ! CHECK-DAG: %[[x:.*]] = fir.load %[[x_decl]]#0 : !fir.ref<f32>

  ! CHECK: fir.if %{{.*}} {
  ! CHECK:   fir.call @_FortranABesselJnX0_4(%{{.*}}, %[[n1]], %[[n2]], {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, !fir.ref<i8>, i32) -> ()
  ! CHECK: } else {
  ! CHECK:   fir.if %{{.*}} {
  ! CHECK:     fir.call @_FortranABesselJn_4(%{{.*}}, %[[n1]], %[[n2]], %[[x]], %{{.*}}, %{{.*}}, {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, f32, f32, f32, !fir.ref<i8>, i32) -> ()
  ! CHECK:   } else {
  ! CHECK:     fir.if %{{.*}} {
  ! CHECK:       fir.call @_FortranABesselJn_4(%{{.*}}, %[[n1]], %[[n2]], %[[x]], %{{.*}}, %{{.*}}, {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, f32, f32, f32, !fir.ref<i8>, i32) -> ()
  ! CHECK:     } else {
  ! CHECK:       fir.call @_FortranABesselJn_4(%{{.*}}, %[[n1]], %[[n2]], %[[x]], %{{.*}}, %{{.*}}, {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, f32, f32, f32, !fir.ref<i8>, i32) -> ()
  ! CHECK:     }
  ! CHECK:   }
  ! CHECK: }
  r = bessel_jn(n1, n2, x)
  ! CHECK: hlfir.assign %{{.*}} to %[[r_decl]]#0 : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
end subroutine test_transformational_real4

! CHECK-LABEL: func.func @_QPtest_transformational_real8
! CHECK-SAME: %[[argx:.*]]: !fir.ref<f64>{{.*}}, %[[argn1:.*]]: !fir.ref<i32>{{.*}}, %[[argn2:.*]]: !fir.ref<i32>{{.*}}, %[[argr:.*]]: !fir.box<!fir.array<?xf64>>{{.*}}
subroutine test_transformational_real8(x, n1, n2, r)
  real(8) :: x
  integer :: n1, n2
  real(8) :: r(:)

  ! CHECK: %[[n1_decl:.*]]:2 = hlfir.declare %[[argn1]] {{.*}}
  ! CHECK: %[[n2_decl:.*]]:2 = hlfir.declare %[[argn2]] {{.*}}
  ! CHECK: %[[r_decl:.*]]:2 = hlfir.declare %[[argr]] {{.*}}
  ! CHECK: %[[x_decl:.*]]:2 = hlfir.declare %[[argx]] {{.*}}
  ! CHECK-DAG: %[[n1:.*]] = fir.load %[[n1_decl]]#0 : !fir.ref<i32>
  ! CHECK-DAG: %[[n2:.*]] = fir.load %[[n2_decl]]#0 : !fir.ref<i32>
  ! CHECK-DAG: %[[x:.*]] = fir.load %[[x_decl]]#0 : !fir.ref<f64>

  ! CHECK: fir.if %{{.*}} {
  ! CHECK:   fir.call @_FortranABesselJnX0_8(%{{.*}}, %[[n1]], %[[n2]], {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, !fir.ref<i8>, i32) -> ()
  ! CHECK: } else {
  ! CHECK:   fir.if %{{.*}} {
  ! CHECK:     fir.call @_FortranABesselJn_8(%{{.*}}, %[[n1]], %[[n2]], %[[x]], %{{.*}}, %{{.*}}, {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, f64, f64, f64, !fir.ref<i8>, i32) -> ()
  ! CHECK:   } else {
  ! CHECK:     fir.if %{{.*}} {
  ! CHECK:       fir.call @_FortranABesselJn_8(%{{.*}}, %[[n1]], %[[n2]], %[[x]], %{{.*}}, %{{.*}}, {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, f64, f64, f64, !fir.ref<i8>, i32) -> ()
  ! CHECK:     } else {
  ! CHECK:       fir.call @_FortranABesselJn_8(%{{.*}}, %[[n1]], %[[n2]], %[[x]], %{{.*}}, %{{.*}}, {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, f64, f64, f64, !fir.ref<i8>, i32) -> ()
  ! CHECK:     }
  ! CHECK:   }
  ! CHECK: }
  r = bessel_jn(n1, n2, x)
  ! CHECK: hlfir.assign %{{.*}} to %[[r_decl]]#0 : !hlfir.expr<?xf64>, !fir.box<!fir.array<?xf64>>
end subroutine test_transformational_real8
