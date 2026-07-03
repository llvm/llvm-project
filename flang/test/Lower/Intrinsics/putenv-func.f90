!RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

!CHECK-LABEL: func.func @_QPputenv_test
!CHECK-SAME:  %[[dummyStr:.*]]: !fir.boxchar<1> {fir.bindc_name = "str"}) -> i32 {
integer function putenv_test(str)
CHARACTER(len=255) :: str

!CHECK-DAG:   %[[func_result:.*]] = fir.alloca i32 {bindc_name = "putenv_test", uniq_name = "_QFputenv_testEputenv_test"}
!CHECK-DAG:   %[[func_result_decl:.*]]:{{.*}} = hlfir.declare %[[func_result]] {uniq_name = "_QFputenv_testEputenv_test"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-DAG:   %[[src_str_addr:.*]] = fir.address_of(@_{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>
!CHECK-DAG:   %[[line_value:.*]] = arith.constant {{.*}} : i64
!CHECK-DAG:   %[[str:.*]] = fir.convert {{.*}} (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
!CHECK-DAG:   %[[str_len:.*]] = fir.convert {{.*}} : (index) -> i64
!CHECK-DAG:   %[[src_str:.*]] = fir.convert %[[src_str_addr]] : (!fir.ref<!fir.char<1,{{.*}}>) -> !fir.ref<i8>
!CHECK-DAG:   %[[line:.*]] = fir.convert %[[line_value]] : (i64) -> i32
!CHECK:       %[[putenv_result:.*]] = fir.call @_FortranAPutEnv(%[[str]], %[[str_len]], %[[src_str]], %[[line]])
!CHECK-SAME:  -> i32

! Check _FortranAPutEnv result code handling
!CHECK-DAG:   hlfir.assign %[[putenv_result]] to %[[func_result_decl]]#0 : i32, !fir.ref<i32>
!CHECK-DAG:   %[[load_result:.*]] = fir.load %[[func_result_decl]]#0 : !fir.ref<i32>
!CHECK:       return %[[load_result]] : i32
putenv_test = putenv(str)
end function putenv_test
