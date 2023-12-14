! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPall_args(
! CHECK-SAME:    %[[commandArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "command"}, 
! CHECK-SAME:    %[[exitvalArg:.*]]: !fir.ref<i32> {fir.bindc_name = "exitval"}) { 
subroutine all_args(command, exitVal)
CHARACTER(*) :: command
INTEGER :: exitVal
call system(command, exitVal)
! CHECK-NEXT:    %[[commandUnbox:.*]]:2 = fir.unboxchar %[[commandArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index) 
! CHECK-NEXT:    %[[commandDeclare:.*]]:2 = hlfir.declare %[[commandUnbox]]#0 typeparams %[[commandUnbox]]#1 {uniq_name = "_QFall_argsEcommand"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK-NEXT:    %[[exitvalDeclare:.*]]:2 = hlfir.declare %[[exitvalArg]] {uniq_name = "_QFall_argsEexitval"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-NEXT:    %[[commandBox:.*]] = fir.embox %[[commandDeclare]]#1 typeparams %[[commandUnbox]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-NEXT:    %[[exitvalBox:.*]] = fir.embox %[[exitvalDeclare]]#1 : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK:         %c9_i32 = arith.constant 9 : i32
! CHECK-NEXT:    %[[command:.*]] = fir.convert %[[commandBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none> 
! CHECK-NEXT:    %[[exitval:.*]] = fir.convert %[[exitvalBox]] : (!fir.box<i32>) -> !fir.box<none>
! CHECK:         %[[VAL_9:.*]] = fir.call @_FortranASystem(%[[command]], %[[exitval]], %[[VAL_8:.*]], %c9_i32) fastmath<contract> : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK-NEXT:    return
! CHECK-NEXT:    }
end subroutine all_args

! CHECK-LABEL: func.func @_QPonly_command(
! CHECK-SAME:    %[[commandArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "command"}) {
subroutine only_command(command)
CHARACTER(*) :: command
call system(command)
! CHECK-NEXT:    %[[commandUnbox:.*]]:2 = fir.unboxchar %[[commandArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT:    %[[commandDeclare:.*]]:2 = hlfir.declare %[[commandUnbox]]#0 typeparams %[[commandUnbox]]#1 {uniq_name = "_QFonly_commandEcommand"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK-NEXT:    %[[commandBox:.*]] = fir.embox %[[commandDeclare]]#1 typeparams %[[commandUnbox]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-NEXT:    %[[absentBox:.*]] = fir.absent !fir.box<none>
! CHECK:         %c27_i32 = arith.constant 27 : i32
! CHECK-NEXT:    %[[command:.*]] = fir.convert %[[commandBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK:         %[[VAL_7:.*]] = fir.call @_FortranASystem(%[[command]], %[[absentBox]], %[[VAL_6:.*]], %c27_i32) fastmath<contract> : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK-NEXT:    return
! CHECK-NEXT:   }
end subroutine only_command
