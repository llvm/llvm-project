! RUN: bbc -emit-fir %s -o - | FileCheck %s --check-prefixes=CHECK%if target=x86_64{{.*}} %{,CHECK-KIND10%}%if flang-supports-f128-math %{,CHECK-KIND16%}

! CHECK-LABEL: func @_QPnorm2_test_4(
real(4) function norm2_test_4(a)
  real(4) :: a(:)
  ! CHECK:  %[[c0:.*]] = arith.constant 0 : index
  ! CHECK: %[[a1:.*]] = fir.declare{{.*}}a"
  ! CHECK: %[[a:.*]] = fir.rebox %[[a1]]{{.*}}
  ! CHECK-DAG:  %[[arr:.*]] = fir.convert %[[a]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
  ! CHECK:  %[[dim:.*]] = fir.convert %[[c0]] : (index) -> i32
  norm2_test_4 = norm2(a)
  ! CHECK:  %{{.*}} = fir.call @_FortranANorm2_4(%[[arr]], %{{.*}}, %{{.*}}, %[[dim]]) {{.*}} : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> f32
end function norm2_test_4

! CHECK-LABEL: func @_QPnorm2_test_8(
real(8) function norm2_test_8(a)
  real(8) :: a(:,:)
  norm2_test_8 = norm2(a)
  ! CHECK: fir.call @_FortranANorm2_8(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> f64
end function norm2_test_8

! CHECK-KIND10-LABEL: func @_QPnorm2_test_10(
function norm2_test_10(a)
  integer, parameter :: kind10 = merge(10, 4, selected_real_kind(p=18).eq.10)
  real(kind10) :: a(:,:,:), norm2_test_10
  norm2_test_10 = norm2(a)
  ! CHECK-KIND10: fir.call @_FortranANorm2_10(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> f80
end function norm2_test_10

! CHECK-KIND16-LABEL: func @_QPnorm2_test_16(
function norm2_test_16(a)
  integer, parameter :: kind16 = merge(16, 4, selected_real_kind(p=33).eq.16)
  real(kind16) :: a(:,:,:), norm2_test_16
  norm2_test_16 = norm2(a)
  ! CHECK-KIND16:  %{{.*}} = fir.call @_FortranANorm2_16(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> f128
end function norm2_test_16

! CHECK-LABEL: func @_QPnorm2_test_dim_2(
subroutine norm2_test_dim_2(a,r)
  real :: a(:,:)
  real :: r(:)
  ! CHECK-DAG:  %[[dim:.*]] = arith.constant 1 : i32
  ! CHECK-DAG: %[[a1:.*]] = fir.declare{{.*}}a"
  ! CHECK-DAG: %[[a:.*]] = fir.rebox %[[a1]]{{.*}}
  ! CHECK-DAG:  %[[r:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
  ! CHECK-DAG:  %[[res:.*]] = fir.convert %[[r]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK:  %[[arr:.*]] = fir.convert %[[a]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
  r = norm2(a,dim=1)
  ! CHECK:  fir.call @_FortranANorm2Dim(%[[res]], %[[arr]], %[[dim]], %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> ()
  ! CHECK-DAG:  fir.freemem
end subroutine norm2_test_dim_2

! CHECK-LABEL: func @_QPnorm2_test_dim_3(
subroutine norm2_test_dim_3(a,r)
  real :: a(:,:,:)
  real :: r(:,:)
  ! CHECK-DAG:  %[[dim:.*]] = arith.constant 3 : i32
  r = norm2(a,dim=3)
  ! CHECK:  fir.call @_FortranANorm2Dim(%{{.*}}, %{{.*}}, %[[dim]], %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> ()
end subroutine norm2_test_dim_3

! CHECK-LABEL: func @_QPnorm2_test_real16(
subroutine norm2_test_real16(a,r)
  integer, parameter :: kind16 = merge(16, 4, selected_real_kind(p=33).eq.16)
  real(kind16) :: a(:,:,:)
  real(kind16) :: r(:,:)
  r = norm2(a,dim=3)
  ! CHECK-KIND16:  fir.call @_FortranANorm2DimReal16(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> ()
  ! CHECK-KIND16:  fir.freemem
end subroutine norm2_test_real16
