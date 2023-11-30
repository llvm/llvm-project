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
! CHECK:         %0 = fir.declare %[[cmdstatArg]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFall_args_optionalEcmdval"} : (!fir.ref<i32>) -> !fir.ref<i32>
! CHECK-NEXT:    %1:2 = fir.unboxchar %[[commandArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT:    %2 = fir.declare %1#0 typeparams %1#1 {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFall_args_optionalEcommand"} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.ref<!fir.char<1,?>>
! CHECK-NEXT:    %3 = fir.emboxchar %2, %1#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK-NEXT:    %4 = fir.declare %[[exitstatArg]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFall_args_optionalEexitval"} : (!fir.ref<i32>) -> !fir.ref<i32>
! CHECK-NEXT:    %5 = fir.declare %[[waitArg]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFall_args_optionalEiswait"} : (!fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>>
! CHECK-NEXT:    %6:2 = fir.unboxchar %[[cmdmsgArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT:    %7 = fir.declare %6#0 typeparams %6#1 {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFall_args_optionalEmsg"} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.ref<!fir.char<1,?>>
! CHECK-NEXT:    %8 = fir.emboxchar %7, %6#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK-NEXT:    %9 = fir.is_present %5 : (!fir.ref<!fir.logical<4>>) -> i1
! CHECK-NEXT:    %10 = fir.is_present %4 : (!fir.ref<i32>) -> i1
! CHECK-NEXT:    %11 = fir.is_present %0 : (!fir.ref<i32>) -> i1
! CHECK-NEXT:    %12 = fir.is_present %8 : (!fir.boxchar<1>) -> i1
! CHECK-NEXT:    %13 = fir.embox %2 typeparams %1#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-NEXT:    %14 = fir.if %9 -> (!fir.logical<4>) {
! CHECK-NEXT:      %31 = fir.load %5 : !fir.ref<!fir.logical<4>>
! CHECK-NEXT:      fir.result %31 : !fir.logical<4>
! CHECK-NEXT:    } else {
! CHECK-NEXT:      %31 = fir.convert %false : (i1) -> !fir.logical<4>
! CHECK-NEXT:      fir.result %31 : !fir.logical<4>
! CHECK-NEXT:    }
! CHECK-NEXT:    %15 = fir.embox %4 : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK-NEXT:    %16 = fir.absent !fir.box<i32>
! CHECK-NEXT:    %17 = arith.select %10, %15, %16 : !fir.box<i32>
! CHECK-NEXT:    %18 = fir.embox %0 : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK-NEXT:    %19 = arith.select %11, %18, %16 : !fir.box<i32>
! CHECK-NEXT:    %20 = fir.embox %7 typeparams %6#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-NEXT:    %21 = fir.absent !fir.box<!fir.char<1,?>> 
! CHECK-NEXT:    %22 = arith.select %12, %20, %21 : !fir.box<!fir.char<1,?>>
! CHECK-NEXT:    %23 = fir.address_of(@_QQclX76c8fd75e0e20222cfcde5fe9055bcbe) : !fir.ref<!fir.char<1,96>>
! CHECK-NEXT:    %24 = fir.convert %13 : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-NEXT:    %25 = fir.convert %14 : (!fir.logical<4>) -> i1
! CHECK-NEXT:    %26 = fir.convert %17 : (!fir.box<i32>) -> !fir.box<none>
! CHECK-NEXT:    %27 = fir.convert %19 : (!fir.box<i32>) -> !fir.box<none>
! CHECK-NEXT:    %28 = fir.convert %22 : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-NEXT:    %29 = fir.convert %23 : (!fir.ref<!fir.char<1,96>>) -> !fir.ref<i8>
! CHECK-NEXT:    %30 = fir.call @_FortranAExecuteCommandLine(%24, %25, %26, %27, %28, %29, %c14_i32) fastmath<contract> : (!fir.box<none>, i1, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK-NEXT:    return
end subroutine all_args_optional
