! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPall_args(
! CHECK-SAME: %[[commandArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "command"},
! CHECK-SAME: %[[waitArg:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "iswait"},
! CHECK-SAME: %[[exitstatArg:.*]]: !fir.ref<i32> {fir.bindc_name = "exitval"},
! CHECK-SAME: %[[cmdstatArg:.*]]: !fir.ref<i32> {fir.bindc_name = "cmdval"},
! CHECK-SAME: %[[cmdmsgArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "msg"}) {
subroutine all_args(command, isWait, exitVal, cmdVal, msg)
CHARACTER(30) :: command, msg
INTEGER :: exitVal, cmdVal
LOGICAL :: isWait
call execute_command_line(command, isWait, exitVal, cmdVal, msg)
! CHECK:             %0 = fir.declare %[[cmdstatArg]] {uniq_name = "_QFall_argsEcmdval"} : (!fir.ref<i32>) -> !fir.ref<i32>
! CHECK-NEXT:        %[[commandUnbox:.*]]:2 = fir.unboxchar %[[commandArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT:        %[[commandCast:.*]] = fir.convert %[[commandUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,30>>
! CHECK-NEXT:        %3 = fir.declare %[[commandCast]] typeparams %c30 {uniq_name = "_QFall_argsEcommand"} : (!fir.ref<!fir.char<1,30>>, index) -> !fir.ref<!fir.char<1,30>>
! CHECK-NEXT:        %4 = fir.declare %[[exitstatArg]] {uniq_name = "_QFall_argsEexitval"} : (!fir.ref<i32>) -> !fir.ref<i32>
! CHECK-NEXT:        %5 = fir.declare %[[waitArg]] {uniq_name = "_QFall_argsEiswait"} : (!fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>>
! CHECK-NEXT:        %[[cmdmsgUnbox:.*]]:2 = fir.unboxchar %[[cmdmsgArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT:        %[[cmdmsgCast:.*]] = fir.convert %[[cmdmsgUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,30>>
! CHECK-NEXT:        %8 = fir.declare %[[cmdmsgCast]] typeparams %c30 {uniq_name = "_QFall_argsEmsg"} : (!fir.ref<!fir.char<1,30>>, index) -> !fir.ref<!fir.char<1,30>>
! CHECK-NEXT:        %[[commandBox:.*]] = fir.embox %3 : (!fir.ref<!fir.char<1,30>>) -> !fir.box<!fir.char<1,30>>
! CHECK-NEXT:        %[[waitLoaded:.*]] = fir.load %5 : !fir.ref<!fir.logical<4>>
! CHECK-NEXT:        %[[exitstatBox:.*]] = fir.embox %4 : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK-NEXT:        %[[cmdstatBox:.*]] = fir.embox %0 : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK-NEXT:        %[[cmdmsgBox:.*]] = fir.embox %8 : (!fir.ref<!fir.char<1,30>>) -> !fir.box<!fir.char<1,30>>
! CHECK:             %[[command:.*]] = fir.convert %[[commandBox]] : (!fir.box<!fir.char<1,30>>) -> !fir.box<none>
! CHECK-NEXT:        %[[wait:.*]] = fir.convert %[[waitLoaded]] : (!fir.logical<4>) -> i1
! CHECK-NEXT:        %[[exitstat:.*]] = fir.convert %[[exitstatBox]] : (!fir.box<i32>) -> !fir.box<none>
! CHECK-NEXT:        %[[cmdstat:.*]] = fir.convert %[[cmdstatBox]] : (!fir.box<i32>) -> !fir.box<none>
! CHECK-NEXT:        %[[cmdmsg:.*]] = fir.convert %[[cmdmsgBox]] : (!fir.box<!fir.char<1,30>>) -> !fir.box<none>
! CHECK:             %21 = fir.call @_FortranAExecuteCommandLine(%[[command]], %[[wait]], %[[exitstat]], %[[cmdstat]], %[[cmdmsg]], %20, %c13_i32) fastmath<contract> : (!fir.box<none>, i1, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK-NEXT:        return
end subroutine all_args

! CHECK-LABEL: func.func @_QPonly_command_default_wait_true(
! CHECK-SAME: %[[cmdArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "command"}) {
subroutine only_command_default_wait_true(command)
CHARACTER(30) :: command
call execute_command_line(command)
! CHECK:          %0:2 = fir.unboxchar %[[cmdArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT:     %1 = fir.convert %0#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,30>>
! CHECK-NEXT:     %2 = fir.declare %1 typeparams %c30 {uniq_name = "_QFonly_command_default_wait_trueEcommand"} : (!fir.ref<!fir.char<1,30>>, index) -> !fir.ref<!fir.char<1,30>>
! CHECK-NEXT:     %3 = fir.embox %2 : (!fir.ref<!fir.char<1,30>>) -> !fir.box<!fir.char<1,30>>
! CHECK-NEXT:     %4 = fir.absent !fir.box<none>
! CHECK:          %6 = fir.convert %3 : (!fir.box<!fir.char<1,30>>) -> !fir.box<none> 
! CHECK:          %8 = fir.call @_FortranAExecuteCommandLine(%6, %true, %4, %4, %4, %7, %c41_i32) fastmath<contract> : (!fir.box<none>, i1, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK-NEXT:     return
end subroutine only_command_default_wait_true
