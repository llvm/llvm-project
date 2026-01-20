!RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

!CHECK-LABEL: func.func @_QPstr_only
!CHECK-SAME:    %[[dummyStr:.*]]: !fir.boxchar<1> {fir.bindc_name = "str"}) {
subroutine str_only(str)
    CHARACTER(len=*) :: str
    !CHECK-DAG:    %[[scope:.*]] = fir.dummy_scope : !fir.dscope
    !CHECK-DAG:    %[[unbox_str:.*]]:2 = fir.unboxchar %[[dummyStr]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    !CHECK-DAG:    %[[str_decl:.*]]:2 = hlfir.declare %[[unbox_str]]#0 typeparams %[[unbox_str]]#1 dummy_scope %[[scope]] arg {{[0-9]+}} {uniq_name = "_QFstr_onlyEstr"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    !CHECK-DAG:    %[[src_str_addr:.*]] = fir.address_of(@_{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
    !CHECK-DAG:    %[[line_value:.*]] = arith.constant {{.*}} : i64
    !CHECK-DAG:    %[[str:.*]] = fir.convert %[[str_decl]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
    !CHECK-DAG:    %[[str_len:.*]] = fir.convert %[[unbox_str]]#1 : (index) -> i64
    !CHECK-DAG:    %[[src_str:.*]] = fir.convert %[[src_str_addr]] : (!fir.ref<!fir.char<1,{{.*}}>) -> !fir.ref<i8>
    !CHECK-DAG:    %[[line:.*]] = fir.convert %[[line_value]] : (i64) -> i32
    !CHECK:        fir.call @_FortranAPutEnv(%[[str]], %[[str_len]], %[[src_str]], %[[line]])
    !CHECK-SAME:   : (!fir.ref<i8>, i64, !fir.ref<i8>, i32)
    !CHECK-SAME:   -> i32
    call putenv(str)
    !CHECK:        return
end subroutine str_only
    !CHECK:         }

    !CHECK-LABEL: func.func @_QPall_arguments
    !CHECK-SAME:    %[[dummyStr:.*]]: !fir.boxchar<1> {fir.bindc_name = "str"}
    !CHECK-SAME:    %[[dummyStat:.*]]: !fir.ref<i32> {fir.bindc_name = "status"}
    !CHECK-SAME:    ) {
subroutine all_arguments(str, status)
    CHARACTER(len=*) :: str
    INTEGER :: status
    !CHECK-DAG:    %[[scope:.*]] = fir.dummy_scope : !fir.dscope
    !CHECK-DAG:    %[[unbox_str:.*]]:2 = fir.unboxchar %[[dummyStr]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    !CHECK-DAG:    %[[str_decl:.*]]:2 = hlfir.declare %[[unbox_str]]#0 typeparams %[[unbox_str]]#1 dummy_scope %[[scope]] arg {{[0-9]+}} {uniq_name = "_QFall_argumentsEstr"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    !CHECK-DAG:    %[[status_decl:.*]]:2 = hlfir.declare %[[dummyStat]] dummy_scope %[[scope]] arg {{[0-9]+}} {uniq_name = "_QFall_argumentsEstatus"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
    !CHECK-DAG:    %[[src_str_addr:.*]] = fir.address_of(@_{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
    !CHECK-DAG:    %[[line_value:.*]] = arith.constant {{.*}} : i64
    !CHECK-DAG:    %[[str:.*]] = fir.convert %[[str_decl]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
    !CHECK-DAG:    %[[str_len:.*]] = fir.convert %[[unbox_str]]#1 : (index) -> i64
    !CHECK-DAG:    %[[src_str:.*]] = fir.convert %[[src_str_addr]] : (!fir.ref<!fir.char<1,{{.*}}>) -> !fir.ref<i8>
    !CHECK-DAG:    %[[line:.*]] = fir.convert %[[line_value]] : (i64) -> i32
    !CHECK:        %[[putenv_result:.*]] = fir.call @_FortranAPutEnv(%[[str]], %[[str_len]], %[[src_str]], %[[line]])
    !CHECK-SAME:   : (!fir.ref<i8>, i64, !fir.ref<i8>, i32)
    !CHECK-SAME:   -> i32

    !CHECK-DAG:    %[[status_i64:.*]] = fir.convert %[[status_decl]]#0 : (!fir.ref<i32>) -> i64
    !CHECK-DAG:    %[[c_null:.*]] = arith.constant 0 : i64
    !CHECK-DAG:    %[[cmp_result:.*]] = arith.cmpi ne, %[[status_i64]], %[[c_null]] : i64
    !CHECK:        fir.if %[[cmp_result]] {
    !CHECK-NEXT:   fir.store %[[putenv_result]] to %[[status_decl]]#0 : !fir.ref<i32>
    !CHECK-NEXT:   }
    call putenv(str, status)
    !CHECK:        return
end subroutine all_arguments
    !CHECK:        }
