!RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

!CHECK-LABEL: func.func @_QPhostnm_only
!CHECK-SAME:    %[[dummyHn:.*]]: !fir.boxchar<1> {fir.bindc_name = "hn"}) {
subroutine hostnm_only(hn)
  CHARACTER(len=255) :: hn

  ! Check that _FortranAHostnm is called with boxed char 255, some other char
  ! string of variable length (source file path) and some integer (source line)
  !CHECK-DAG:     %[[line:.*]] = arith.constant {{.*}} : i32
  !CHECK-DAG:     %[[hn:.*]] = fir.convert {{.*}} (!fir.box<!fir.char<1,255>>) -> !fir.box<none>
  !CHECK-DAG:     %[[src_path:.*]] = fir.convert {{.*}} (!fir.ref<!fir.char<1,{{.*}} -> !fir.ref<i8>
  !CHECK: fir.call @_FortranAHostnm(%[[hn]], %[[src_path]], %[[line]])
  !CHECK-SAME:    -> i32
  call hostnm(hn)
end subroutine hostnm_only

!CHECK-LABEL: func.func @_QPall_arguments
!CHECK-SAME:    %[[dummyHn:.*]]: !fir.boxchar<1> {fir.bindc_name = "hn"},
!CHECK-SAME:    %[[dummyStat:.*]]: !fir.ref<i32> {fir.bindc_name = "status"}) {
subroutine all_arguments(hn, status)
  CHARACTER(len=255) :: hn
  INTEGER :: status

  ! Check that _FortranAHostnm is called with boxed char 255, some other char
  ! string of variable length (source file path) and some integer (source line)
  !CHECK-DAG: %[[line:.*]] = arith.constant {{.*}} : i32
  !CHECK-DAG: %[[hn:.*]] = fir.convert {{.*}} (!fir.box<!fir.char<1,255>>) -> !fir.box<none>
  !CHECK-DAG: %[[src_path:.*]] = fir.convert {{.*}} (!fir.ref<!fir.char<1,{{.*}} -> !fir.ref<i8>
  !CHECK:     %[[hn_result:.*]] = fir.call @_FortranAHostnm(%[[hn]], %[[src_path]], %[[line]])
  !CHECK-SAME:    -> i32

  ! Check _FortranAHostnm result code handling
  !CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
  !CHECK-DAG: %[[cmp_result:.*]] = arith.cmpi ne, {{.*}}, %[[c0_i64]] : i64
  !CHECK:     fir.store %[[hn_result]] {{.*}} !fir.ref<i32>
  call hostnm(hn, status)
end subroutine all_arguments
