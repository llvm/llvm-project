!RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

!CHECK-LABEL: func.func @_QPhostnm_test
!CHECK-SAME:    %[[dummyHn:.*]]: !fir.boxchar<1> {fir.bindc_name = "hn"}) -> i32 {
integer function hostnm_test(hn)
  CHARACTER(len=255) :: hn

  ! Check that _FortranAHostnm is called with boxed char 255, some other char
  ! string of variable length (source file path) and some integer (source line)
  !CHECK-DAG:   %[[func_result:.*]] = fir.alloca i32 {bindc_name = "hostnm_test", uniq_name = "_QFhostnm_testEhostnm_test"}
  !CHECK-DAG:   %[[func_result_decl:.*]]:{{.*}} = hlfir.declare %[[func_result]] {uniq_name = "_QFhostnm_testEhostnm_test"} : {{.*}}fir.ref<i32>{{.*}}
  !CHECK-DAG:   %[[line:.*]] = arith.constant {{.*}} : i32
  !CHECK-DAG:   %[[hn:.*]] = fir.convert {{.*}} (!fir.box<!fir.char<1,255>>) -> !fir.box<none>
  !CHECK-DAG:   %[[src_path:.*]] = fir.convert {{.*}} (!fir.ref<!fir.char<1,{{.*}} -> !fir.ref<i8>
  !CHECK:     %[[hn_result:.*]] = fir.call @_FortranAHostnm(%[[hn]], %[[src_path]], %[[line]])
  !CHECK-SAME:  -> i32

  ! Check _FortranAHostnm result code handling
  !CHECK-DAG:   hlfir.assign %[[hn_result]] to %[[func_result_decl]]{{.*}}i32{{.*}}
  !CHECK-DAG:   %[[load_result:.*]] = fir.load %[[func_result_decl]]{{.*}}i32{{.*}}
  !CHECK:     return %[[load_result]] : i32
  hostnm_test = hostnm(hn)
end function hostnm_test
