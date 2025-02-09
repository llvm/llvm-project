! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPall_args_optional(
! CHECK-SAME: %[[commandArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "command", fir.optional},
! CHECK-SAME: %[[waitArg:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "iswait", fir.optional},
! CHECK-SAME: %[[exitstatArg:.*]]: !fir.ref<i32> {fir.bindc_name = "exitval", fir.optional},
! CHECK-SAME: %[[cmdstatArg:.*]]: !fir.ref<i32> {fir.bindc_name = "cmdval", fir.optional},
! CHECK-SAME: %[[cmdmsgArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "msg", fir.optional}) {
subroutine all_args_optional(command, isWait, exitVal, cmdVal, msg)
  CHARACTER(*), OPTIONAL :: command, msg
  INTEGER, OPTIONAL :: exitVal, cmdVal
  LOGICAL, OPTIONAL :: isWait
  ! Note: command is not optional in execute_command_line and must be present
  call execute_command_line(command, isWait, exitVal, cmdVal, msg)
! CHECK-NEXT:    %[[c14:.*]] = arith.constant 14 : i32 
! CHECK-NEXT:    %true = arith.constant true 
! CHECK-NEXT:    %[[c0:.*]] = arith.constant 0 : i64 
! CHECK-NEXT:    %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK-NEXT:    %[[cmdstatDeclare:.*]] = fir.declare %[[cmdstatArg]] dummy_scope %[[DSCOPE]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFall_args_optionalEcmdval"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
! CHECK-NEXT:    %[[commandUnbox:.*]]:2 = fir.unboxchar %[[commandArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT:    %[[commandDeclare:.*]] = fir.declare %[[commandUnbox]]#0 typeparams %[[commandUnbox]]#1 dummy_scope %[[DSCOPE]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFall_args_optionalEcommand"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> !fir.ref<!fir.char<1,?>>
! CHECK-NEXT:    %[[commandBoxTemp:.*]] = fir.emboxchar %[[commandDeclare]], %[[commandUnbox]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK-NEXT:    %[[exitstatDeclare:.*]] = fir.declare %[[exitstatArg]] dummy_scope %[[DSCOPE]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFall_args_optionalEexitval"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
! CHECK-NEXT:    %[[waitDeclare:.*]] = fir.declare %[[waitArg]] dummy_scope %[[DSCOPE]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFall_args_optionalEiswait"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> !fir.ref<!fir.logical<4>>
! CHECK-NEXT:    %[[cmdmsgUnbox:.*]]:2 = fir.unboxchar %[[cmdmsgArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT:    %[[cmdmsgDeclare:.*]] = fir.declare %[[cmdmsgUnbox]]#0 typeparams %[[cmdmsgUnbox]]#1 dummy_scope %[[DSCOPE]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFall_args_optionalEmsg"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> !fir.ref<!fir.char<1,?>>
! CHECK-NEXT:    %[[cmdmsgBoxTemp:.*]] = fir.emboxchar %[[cmdmsgDeclare]], %[[cmdmsgUnbox]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK-NEXT:    %[[exitstatIsPresent:.*]] = fir.is_present %[[exitstatDeclare]] : (!fir.ref<i32>) -> i1
! CHECK-NEXT:    %[[cmdstatIsPresent:.*]] = fir.is_present %[[cmdstatDeclare]] : (!fir.ref<i32>) -> i1
! CHECK-NEXT:    %[[cmdmsgIsPresent:.*]] = fir.is_present %[[cmdmsgBoxTemp]] : (!fir.boxchar<1>) -> i1
! CHECK-NEXT:    %[[commandBox:.*]] = fir.embox %[[commandDeclare]] typeparams %[[commandUnbox]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-NEXT:    %[[exitstatArgBox:.*]] = fir.embox %[[exitstatDeclare]] : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK-NEXT:    %[[absentBoxi32:.*]] = fir.absent !fir.box<i32>
! CHECK-NEXT:    %[[exitstatBox:.*]] = arith.select %[[exitstatIsPresent]], %[[exitstatArgBox]], %[[absentBoxi32]] : !fir.box<i32>
! CHECK-NEXT:    %[[cmdstatArgBox:.*]] = fir.embox %[[cmdstatDeclare]] : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK-NEXT:    %[[cmdstatBox:.*]] = arith.select %[[cmdstatIsPresent]], %[[cmdstatArgBox]], %[[absentBoxi32]] : !fir.box<i32>
! CHECK-NEXT:    %[[cmdmsgArgBox:.*]] = fir.embox %[[cmdmsgDeclare]] typeparams %[[cmdmsgUnbox]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-NEXT:    %[[absentBox:.*]] = fir.absent !fir.box<!fir.char<1,?>> 
! CHECK-NEXT:    %[[cmdmsgBox:.*]] = arith.select %[[cmdmsgIsPresent]], %[[cmdmsgArgBox]], %[[absentBox]] : !fir.box<!fir.char<1,?>>
! CHECK-NEXT:    %[[waitCast:.*]] = fir.convert %[[waitDeclare]]  : (!fir.ref<!fir.logical<4>>) -> i64
! CHECK-NEXT:    %[[waitPresent:.*]] = arith.cmpi ne, %[[waitCast]], %[[c0]] : i64
! CHECK-NEXT:    %[[wait:.*]] = fir.if %[[waitPresent]] -> (i1) {
! CHECK-NEXT:      %[[waitLoaded:.*]] = fir.load %[[waitDeclare]] : !fir.ref<!fir.logical<4>>
! CHECK-NEXT:      %[[waitTrueVal:.*]] = fir.convert %[[waitLoaded]] : (!fir.logical<4>) -> i1
! CHECK-NEXT:      fir.result %[[waitTrueVal]] : i1
! CHECK-NEXT:    } else {
! CHECK-NEXT:      fir.result %true : i1
! CHECK-NEXT:    }
! CHECK:         %[[command:.*]] = fir.convert %[[commandBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-NEXT:    %[[exitstat:.*]] = fir.convert %[[exitstatBox]] : (!fir.box<i32>) -> !fir.box<none>
! CHECK-NEXT:    %[[cmdstat:.*]] = fir.convert %[[cmdstatBox]] : (!fir.box<i32>) -> !fir.box<none>
! CHECK-NEXT:    %[[cmdmsg:.*]] = fir.convert %[[cmdmsgBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK:         fir.call @_FortranAExecuteCommandLine(%[[command]], %[[wait]], %[[exitstat]], %[[cmdstat]], %[[cmdmsg]], %[[VAL_29:.*]], %[[c14]]) fastmath<contract> : (!fir.box<none>, i1, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
! CHECK-NEXT:    return
end subroutine all_args_optional
