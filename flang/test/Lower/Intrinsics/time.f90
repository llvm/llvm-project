!RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

!CHECK-LABEL: func.func @_QPtime_test() -> i64
function time_test()
  Integer(kind=8) :: time_test


  !CHECK-DAG:   %[[func_result:.*]] = fir.alloca i64 {bindc_name = "time_test", uniq_name = "_QFtime_testEtime_test"}
  !CHECK-DAG:   %[[func_result_decl:.*]]:{{.*}} = hlfir.declare %[[func_result]] {uniq_name = "_QFtime_testEtime_test"} : {{.*}}fir.ref<i64>{{.*}}
  !CHECK:     %[[call_result:.*]] = fir.call @_FortranAtime()
  !CHECK-SAME:  -> i64

  !CHECK-DAG:   hlfir.assign %[[call_result]] to %[[func_result_decl]]#0 : i64, !fir.ref<i64>
  !CHECK-DAG:   %[[load_result:.*]] = fir.load %[[func_result_decl]]#0 : !fir.ref<i64>
  !CHECK:     return %[[load_result]] : i64
  time_test = time()
end function time_test
