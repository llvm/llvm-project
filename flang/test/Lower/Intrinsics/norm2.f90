! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPnorm2_test_4(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}) -> f32
real(4) function norm2_test_4(a)
  real(4) :: a(:)
  ! CHECK-DAG:  %[[c0:.*]] = arith.constant 0 : index
  ! CHECK-DAG:  %[[arr:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
  ! CHECK:  %[[dim:.*]] = fir.convert %[[c0]] : (index) -> i32
  norm2_test_4 = norm2(a)
  ! CHECK:  %{{.*}} = fir.call @_FortranANorm2_4(%[[arr]], %{{.*}}, %{{.*}}, %[[dim]]) {{.*}} : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> f32
end function norm2_test_4

! CHECK-LABEL: func @_QPnorm2_test_8(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xf64>>{{.*}}) -> f64
real(8) function norm2_test_8(a)
  real(8) :: a(:,:)
  ! CHECK-DAG:  %[[c0:.*]] = arith.constant 0 : index
  ! CHECK-DAG:  %[[arr:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?xf64>>) -> !fir.box<none>
  ! CHECK:  %[[dim:.*]] = fir.convert %[[c0]] : (index) -> i32
  norm2_test_8 = norm2(a)
  ! CHECK:  %{{.*}} = fir.call @_FortranANorm2_8(%[[arr]], %{{.*}}, %{{.*}}, %[[dim]]) {{.*}} : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> f64
end function norm2_test_8

! CHECK-LABEL: func @_QPnorm2_test_10(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x?xf80>>{{.*}}) -> f80
real(10) function norm2_test_10(a)
  real(10) :: a(:,:,:)
  ! CHECK-DAG:  %[[c0:.*]] = arith.constant 0 : index
  ! CHECK-DAG:  %[[arr:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?x?xf80>>) -> !fir.box<none>
  ! CHECK:  %[[dim:.*]] = fir.convert %[[c0]] : (index) -> i32
  norm2_test_10 = norm2(a)
  ! CHECK:  %{{.*}} = fir.call @_FortranANorm2_10(%[[arr]], %{{.*}}, %{{.*}}, %[[dim]]) {{.*}} : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> f80
end function norm2_test_10

! CHECK-LABEL: func @_QPnorm2_test_16(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x?xf128>>{{.*}}) -> f128
real(16) function norm2_test_16(a)
  real(16) :: a(:,:,:)
  ! CHECK-DAG:  %[[c0:.*]] = arith.constant 0 : index
  ! CHECK-DAG:  %[[arr:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?x?xf128>>) -> !fir.box<none>
  ! CHECK:  %[[dim:.*]] = fir.convert %[[c0]] : (index) -> i32
  norm2_test_16 = norm2(a)
  ! CHECK:  %{{.*}} = fir.call @_FortranANorm2_16(%[[arr]], %{{.*}}, %{{.*}}, %[[dim]]) {{.*}} : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> f128
end function norm2_test_16

! CHECK-LABEL: func @_QPnorm2_test_dim_2(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xf32>>{{.*}}, %[[arg1:.*]]: !fir.box<!fir.array<?xf32>>{{.*}})
subroutine norm2_test_dim_2(a,r)
  real :: a(:,:)
  real :: r(:)
  ! CHECK-DAG:  %[[dim:.*]] = arith.constant 1 : i32
  ! CHECK-DAG:  %[[r:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
  ! CHECK-DAG:  %[[res:.*]] = fir.convert %[[r]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK:  %[[arr:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
  r = norm2(a,dim=1)
  ! CHECK:  %{{.*}} = fir.call @_FortranANorm2Dim(%[[res]], %[[arr]], %[[dim]], %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
  ! CHECK:  %[[box:.*]] = fir.load %[[r]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK-DAG:  %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK-DAG:  fir.freemem %[[addr]]
end subroutine norm2_test_dim_2

! CHECK-LABEL: func @_QPnorm2_test_dim_3(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x?xf32>>{{.*}}, %[[arg1:.*]]: !fir.box<!fir.array<?x?xf32>>{{.*}})
subroutine norm2_test_dim_3(a,r)
  real :: a(:,:,:)
  real :: r(:,:)
  ! CHECK-DAG:  %[[dim:.*]] = arith.constant 3 : i32
  ! CHECK-DAG:  %[[r:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xf32>>>
  ! CHECK-DAG:  %[[res:.*]] = fir.convert %[[r]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK:  %[[arr:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?x?xf32>>) -> !fir.box<none>
  r = norm2(a,dim=3)
  ! CHECK:  %{{.*}} = fir.call @_FortranANorm2Dim(%[[res]], %[[arr]], %[[dim]], %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
  ! CHECK:  %[[box:.*]] = fir.load %[[r]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  ! CHECK-DAG:  %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.heap<!fir.array<?x?xf32>>
  ! CHECK-DAG:  fir.freemem %[[addr]]
end subroutine norm2_test_dim_3

! CHECK-LABEL: func @_QPnorm2_test_real16(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x?xf128>>{{.*}}, %[[arg1:.*]]: !fir.box<!fir.array<?x?xf128>>{{.*}})
subroutine norm2_test_real16(a,r)
  real(16) :: a(:,:,:)
  real(16) :: r(:,:)
  ! CHECK-DAG:  %[[dim:.*]] = arith.constant 3 : i32
  ! CHECK-DAG:  %[[r:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xf128>>>
  ! CHECK-DAG:  %[[res:.*]] = fir.convert %[[r]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf128>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK:  %[[arr:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?x?xf128>>) -> !fir.box<none>
  r = norm2(a,dim=3)
  ! CHECK:  %{{.*}} = fir.call @_FortranANorm2DimReal16(%[[res]], %[[arr]], %[[dim]], %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
  ! CHECK:  %[[box:.*]] = fir.load %[[r]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf128>>>>
  ! CHECK-DAG:  %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?x?xf128>>>) -> !fir.heap<!fir.array<?x?xf128>>
  ! CHECK-DAG:  fir.freemem %[[addr]]
end subroutine norm2_test_real16
