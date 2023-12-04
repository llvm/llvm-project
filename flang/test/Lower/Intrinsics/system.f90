! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPall_args() {
! CHECK:         %0 = fir.alloca !fir.char<1,30> {bindc_name = "command", uniq_name = "_QFall_argsEcommand"}
! CHECK-NEXT:    %1:2 = hlfir.declare %0 typeparams %c30 {uniq_name = "_QFall_argsEcommand"} : (!fir.ref<!fir.char<1,30>>, index) -> (!fir.ref<!fir.char<1,30>>, !fir.ref<!fir.char<1,30>>)
! CHECK-NEXT:    %2 = fir.alloca i32 {bindc_name = "exitval", uniq_name = "_QFall_argsEexitval"}
! CHECK-NEXT:    %3:2 = hlfir.declare %2 {uniq_name = "_QFall_argsEexitval"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-NEXT:    %4 = fir.embox %1#1 : (!fir.ref<!fir.char<1,30>>) -> !fir.box<!fir.char<1,30>>
! CHECK-NEXT:    %5 = fir.embox %3#1 : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK-NEXT:    %6 = fir.address_of(@_QQclX6c088457724b07fb671a04391501cd9d) : !fir.ref<!fir.char<1,73>>
! CHECK-NEXT:    %c21_i32 = arith.constant 21 : i32
! CHECK-NEXT:    %7 = fir.convert %4 : (!fir.box<!fir.char<1,30>>) -> !fir.box<none>
! CHECK-NEXT:    %8 = fir.convert %5 : (!fir.box<i32>) -> !fir.box<none>
! CHECK-NEXT:    %9 = fir.convert %6 : (!fir.ref<!fir.char<1,73>>) -> !fir.ref<i8>
! CHECK-NEXT:    %10 = fir.call @_FortranASystem(%7, %8, %9, %c21_i32) fastmath<contract> : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK-NEXT:    return
! CHECK-NEXT:    }
subroutine all_args()
CHARACTER(30) :: command
INTEGER :: exitVal
call system(command, exitVal)
end subroutine all_args

! CHECK-LABEL: func.func @_QPonly_command() {
! CHECK:         %0 = fir.alloca !fir.char<1,30> {bindc_name = "command", uniq_name = "_QFonly_commandEcommand"}
! CHECK-NEXT:    %1:2 = hlfir.declare %0 typeparams %c30 {uniq_name = "_QFonly_commandEcommand"} : (!fir.ref<!fir.char<1,30>>, index) -> (!fir.ref<!fir.char<1,30>>, !fir.ref<!fir.char<1,30>>)
! CHECK-NEXT:    %2 = fir.embox %1#1 : (!fir.ref<!fir.char<1,30>>) -> !fir.box<!fir.char<1,30>>
! CHECK-NEXT:    %3 = fir.absent !fir.box<none>
! CHECK-NEXT:    %4 = fir.address_of(@_QQclX6c088457724b07fb671a04391501cd9d) : !fir.ref<!fir.char<1,73>>
! CHECK-NEXT:    %c38_i32 = arith.constant 38 : i32
! CHECK-NEXT:    %5 = fir.convert %2 : (!fir.box<!fir.char<1,30>>) -> !fir.box<none> 
! CHECK-NEXT:    %6 = fir.convert %4 : (!fir.ref<!fir.char<1,73>>) -> !fir.ref<i8> 
! CHECK-NEXT:    %7 = fir.call @_FortranASystem(%5, %3, %6, %c38_i32) fastmath<contract> : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK-NEXT:    return
! CHECK-NEXT:   }
subroutine only_command()
CHARACTER(30) :: command
call system(command)
end subroutine only_command
