! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPall_args(
! CHECK-SAME:    %[[commandArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "command", fir.optional},
! CHECK-SAME:    %[[exitvalArg:.*]]: !fir.ref<i32> {fir.bindc_name = "exitval", fir.optional}) {
subroutine all_args(command, exitVal)
CHARACTER(*), OPTIONAL :: command
INTEGER, OPTIONAL :: exitVal
call system(command, exitVal)
! CHECK-NEXT:    %[[commandUnbox:.*]]:2 = fir.unboxchar %[[commandArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index) 
! CHECK-NEXT:    %[[commandDeclare:.*]]:2 = hlfir.declare %[[commandUnbox]]#0 typeparams %[[commandUnbox]]#1 {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFall_argsEcommand"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>) 
! CHECK-NEXT:    %[[exitvalDeclare:.*]]:2 = hlfir.declare %[[exitvalArg]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFall_argsEexitval"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>) 
! CHECK-NEXT:    %[[exitvalIsPresent:.*]] = fir.is_present %[[exitvalDeclare]]#0 : (!fir.ref<i32>) -> i1
! CHECK-NEXT:    %[[commandBox:.*]] = fir.embox %[[commandDeclare]]#1 typeparams %[[commandUnbox]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-NEXT:    %[[exitvalBox:.*]] = fir.embox %[[exitvalDeclare]]#1 : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK-NEXT:    %[[absentBox:.*]] = fir.absent !fir.box<i32>
! CHECK-NEXT:    %[[exitvalSelect:.*]] = arith.select %[[exitvalIsPresent]], %[[exitvalBox]], %[[absentBox]] : !fir.box<i32>
! CHECK:         %c9_i32 = arith.constant 9 : i32
! CHECK-NEXT:    %[[command:.*]] = fir.convert %[[commandBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-NEXT:    %[[exitval:.*]] = fir.convert %[[exitvalSelect]] : (!fir.box<i32>) -> !fir.box<none>
! CHECK:         %[[VAL_12:.*]] = fir.call @_FortranASystem(%[[command]], %[[exitval]], %[[VAL_11:.*]], %c9_i32) fastmath<contract> : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK-NEXT:    return
! CHECK-NEXT:    }
end subroutine all_args
