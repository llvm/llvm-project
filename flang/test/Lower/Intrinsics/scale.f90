! RUN: bbc -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK%if target=x86_64{{.*}} %{,CHECK-KIND10%}%if flang-supports-f128-math %{,CHECK-KIND16%}

! CHECK-LABEL: scale_test1
subroutine scale_test1(x, i)
    real :: x, res
  ! CHECK: %[[i:.*]]:2 = hlfir.declare{{.*}}i"
  ! CHECK: %[[res:.*]]:2 = hlfir.declare{{.*}}res"
  ! CHECK: %[[x:.*]]:2 = hlfir.declare{{.*}}x"
  ! CHECK: %[[x_val:.*]] = fir.load %[[x]]#0 : !fir.ref<f32>
    integer :: i
  ! CHECK: %[[i_val:.*]] = fir.load %[[i]]#0 : !fir.ref<i32>
    res = scale(x, i)
  ! CHECK: %[[i_cast:.*]] = fir.convert %[[i_val]] : (i32) -> i64
  ! CHECK: %[[tmp:.*]] = fir.call @_FortranAScale4(%[[x_val]], %[[i_cast]]) {{.*}}: (f32, i64) -> f32
  ! CHECK: hlfir.assign %[[tmp]] to %[[res]]#0 : f32, !fir.ref<f32>
end subroutine scale_test1
  
! CHECK-LABEL: scale_test2
subroutine scale_test2(x, i)
  real(kind=8) :: x, res
  integer :: i
  res = scale(x, i)
! CHECK: fir.call @_FortranAScale8(%{{.*}}, %{{.*}}) {{.*}}: (f64, i64) -> f64
end subroutine scale_test2

! CHECK-KIND10-LABEL: scale_test3
subroutine scale_test3(x, i)
  integer, parameter :: kind10 = merge(10, 4, selected_real_kind(p=18).eq.10)
  real(kind=kind10) :: x, res
  integer :: i
  res = scale(x, i)
! CHECK-KIND10: fir.call @_FortranAScale10(%{{.*}}, %{{.*}}) {{.*}}: (f80, i64) -> f80
end subroutine scale_test3

! CHECK-KIND16-LABEL: scale_test4
subroutine scale_test4(x, i)
  integer, parameter :: kind16 = merge(16, 4, selected_real_kind(p=33).eq.16)
  real(kind=kind16) :: x, res
  integer :: i
  res = scale(x, i)
! CHECK-KIND16: fir.call @_FortranAScale16(%{{.*}}, %{{.*}}) {{.*}}: (f128, i64) -> f128
end subroutine scale_test4
