! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPall_args(
! CHECK-SAME:    %[[commandArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "command"}, 
! CHECK-SAME:    %[[exitstatArg:.*]]: !fir.ref<i32> {fir.bindc_name = "exitstat"}) { 
subroutine all_args(command, exitstat)
CHARACTER(*) :: command
INTEGER :: exitstat
call system(command, exitstat)
! CHECK-NEXT:   %[[cmdstatVal:.*]] = fir.alloca i16
! CHECK-NEXT:   %[[commandUnbox:.*]]:2 = fir.unboxchar %[[commandArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT:   %[[commandDeclare:.*]]:2 = hlfir.declare %[[commandUnbox]]#0 typeparams %[[commandUnbox]]#1 {uniq_name = "_QFall_argsEcommand"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK-NEXT:   %[[exitstatDeclare:.*]]:2 = hlfir.declare %[[exitstatArg]] {uniq_name = "_QFall_argsEexitstat"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-NEXT:   %[[commandBox:.*]] = fir.embox %[[commandDeclare]]#1 typeparams %[[commandUnbox]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-NEXT:   %[[exitstatBox:.*]] = fir.embox %[[exitstatDeclare]]#1 : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK-NEXT:   %[[true:.*]] = arith.constant true
! CHECK-NEXT:   %[[c0_i2:.*]] = arith.constant 0 : i2
! CHECK-NEXT:   %[[c0_i16:.*]] = fir.convert %[[c0_i2]] : (i2) -> i16
! CHECK-NEXT:   fir.store %[[c0_i16]] to %[[cmdstatVal]] : !fir.ref<i16>
! CHECK-NEXT:   %[[cmdstatBox:.*]] = fir.embox %[[cmdstatVal]] : (!fir.ref<i16>) -> !fir.box<i16>
! CHECK-NEXT:   %[[absentBox:.*]] = fir.absent !fir.box<none>
! CHECK:        %[[c9_i32:.*]] = arith.constant 9 : i32
! CHECK-NEXT:   %[[command:.*]] = fir.convert %[[commandBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-NEXT:   %[[exitstat:.*]] = fir.convert %[[exitstatBox]] : (!fir.box<i32>) -> !fir.box<none>
! CHECK-NEXT:   %[[cmdstat:.*]] = fir.convert %[[cmdstatBox]] : (!fir.box<i16>) -> !fir.box<none>
! CHECK:        %[[VAL_13:.*]] = fir.call @_FortranAExecuteCommandLine(%[[command]], %[[true]], %[[exitstat]], %[[cmdstat]], %[[absentBox]], %[[VAL_12:.*]], %[[c9_i32]]) fastmath<contract> : (!fir.box<none>, i1, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK-NEXT:   return
! CHECK-NEXT:  }
end subroutine all_args

! CHECK-LABEL: func.func @_QPonly_command(
! CHECK-SAME:    %[[commandArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "command"}) {
subroutine only_command(command)
CHARACTER(*) :: command
call system(command)
! CHECK-NEXT:   %[[cmdstatVal:.*]] = fir.alloca i16
! CHECK-NEXT:   %[[commandUnbox:.*]]:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT:   %[[commandDeclare:.*]]:2 = hlfir.declare %[[commandUnbox]]#0 typeparams %[[commandUnbox]]#1 {uniq_name = "_QFonly_commandEcommand"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK-NEXT:   %[[commandBox:.*]] = fir.embox %[[commandDeclare]]#1 typeparams %[[commandUnbox]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-NEXT:   %[[true:.*]] = arith.constant true
! CHECK-NEXT:   %[[absentBox:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT:   %[[c0_i2:.*]] = arith.constant 0 : i2
! CHECK-NEXT:   %[[c0_i16:.*]] = fir.convert %[[c0_i2]] : (i2) -> i16
! CHECK-NEXT:   fir.store %[[c0_i16]] to %[[cmdstatVal]] : !fir.ref<i16>
! CHECK-NEXT:   %[[cmdstatBox:.*]] = fir.embox %[[cmdstatVal]] : (!fir.ref<i16>) -> !fir.box<i16>
! CHECK-NEXT:   %[[absentBox2:.*]] = fir.absent !fir.box<none>
! CHECK:        %[[c35_i32:.*]] = arith.constant 35 : i32
! CHECK-NEXT:   %[[command:.*]] = fir.convert %[[commandBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-NEXT:   %[[cmdstat:.*]] = fir.convert %[[cmdstatBox]] : (!fir.box<i16>) -> !fir.box<none>
! CHECK:        %[[VAL_12:.*]] = fir.call @_FortranAExecuteCommandLine(%[[command]], %[[true]], %[[absentBox]], %[[cmdstat]], %[[absentBox2]], %[[VAL_11:.*]], %[[c35_i32]]) fastmath<contract> : (!fir.box<none>, i1, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK-NEXT:   return
! CHECK-NEXT:    }
end subroutine only_command
