!RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

!CHECK-LABEL: func.func @_QPunlink_test
!CHECK-SAME:  %[[dummyPath:.*]]: !fir.boxchar<1> {fir.bindc_name = "path"}) -> i32 {
integer function unlink_test(path)
CHARACTER(len=255) :: path

!CHECK-DAG:   %[[func_result:.*]] = fir.alloca i32 {bindc_name = "unlink_test", uniq_name = "_QFunlink_testEunlink_test"}
!CHECK-DAG:   %[[func_result_decl:.*]]:{{.*}} = hlfir.declare %[[func_result]] {uniq_name = "_QFunlink_testEunlink_test"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-DAG:   %[[src_path_addr:.*]] = fir.address_of(@_{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>
!CHECK-DAG:   %[[line_value:.*]] = arith.constant {{.*}} : i64
!CHECK-DAG:   %[[path:.*]] = fir.convert {{.*}} (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
!CHECK-DAG:   %[[path_len:.*]] = fir.convert {{.*}} : (index) -> i64
!CHECK-DAG:   %[[src_path:.*]] = fir.convert %[[src_path_addr]] : (!fir.ref<!fir.char<1,{{.*}}>) -> !fir.ref<i8>
!CHECK-DAG:   %[[line:.*]] = fir.convert %[[line_value]] : (i64) -> i32
!CHECK:       %[[unlink_result:.*]] = fir.call @_FortranAUnlink(%[[path]], %[[path_len]], %[[src_path]], %[[line]])
!CHECK-SAME:  -> i32

! Check _FortranAUnlink result code handling
!CHECK-DAG:   hlfir.assign %[[unlink_result]] to %[[func_result_decl]]#0 : i32, !fir.ref<i32>
!CHECK-DAG:   %[[load_result:.*]] = fir.load %[[func_result_decl]]#0 : !fir.ref<i32>
!CHECK:       return %[[load_result]] : i32
unlink_test = unlink(path)
end function unlink_test
