! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes=CHECK,CHECK-32 -DDEFAULT_INTEGER_SIZE=32 %s
! RUN: flang-new -fc1 -fdefault-integer-8 -emit-fir %s -o - | FileCheck --check-prefixes=CHECK,CHECK-64 -DDEFAULT_INTEGER_SIZE=64 %s

! CHECK-LABEL: func @_QPcommand_only(
! CHECK-SAME: %[[commandArg:.*]]: !fir.boxchar<1> {fir.bindc_command = "command"}) {
subroutine command_only(command)
    character(len=32) :: command
    call execute_command_line(command)
! CHECK-NOT: fir.call @_FortranAGetEnvVariable
! CHECK-NEXT: return
end subroutine command_only

! CHECK-LABEL: func @_QPcommand_and_wait_only(
! CHECK-SAME: %[[commandArg:.*]]: !fir.boxchar<1> {fir.bindc_command = "command"},
! CHECK-SAME: %[[waitArg:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_command = "wait", fir.optional},
subroutine command_and_wait_only(command, wait)
    character(len=32) :: command, wait
    call execute_command_line(command, wait)
! CHECK: %[[commandUnbox:.*]]:2 = fir.unboxchar %[[commandArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[commandCast:.*]] = fir.convert %[[commandUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK-NEXT: %[[waitUnbox:.*]]:2 = fir.unboxchar %[[waitArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[waitCast:.*]] = fir.convert %[[waitUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK-NEXT: %[[commandBox:.*]] = fir.embox %[[commandCast]] : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK-NEXT: %[[waitBox:.*]] = fir.embox %[[waitCast]] : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK-NEXT: %true = arith.constant true
! CHECK-NEXT: %[[exitstat:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[cmdmsg:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[sourceFileString:.*]] = fir.address_of(@_QQcl{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileexitstat:.*]]>>
! CHECK-NEXT: %[[sourceLine:.*]] = arith.constant [[# @LINE - 11]] : i32
! CHECK-NEXT: %[[command:.*]] = fir.convert %[[commandBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK-NEXT: %[[wait:.*]] = fir.convert %[[waitBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK-NEXT: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileexitstat]]>>) -> !fir.ref<i8>
! CHECK-NEXT: %{{[0-9]+}} = fir.call @_FortranAGetEnvVariable(%[[command]], %[[wait]], %[[exitstat]], %true, %[[cmdmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-NEXT: return
end subroutine command_and_wait_only

! CHECK-LABEL: func @_QPcommand_and_exitstat_only(
! CHECK-SAME: %[[commandArg:.*]]: !fir.boxchar<1> {fir.bindc_command = "command"},
! CHECK-SAME: %[[exitstatArg:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]> {fir.bindc_command = "exitstat"}) {
subroutine command_and_exitstat_only(command, exitstat)
    character(len=32) :: command
    integer :: exitstat
    call execute_command_line(command, EXITSTAT=exitstat)
! CHECK: %[[commandUnbox:.*]]:2 = fir.unboxchar %[[commandArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[commandCast:.*]] = fir.convert %[[commandUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK-NEXT: %[[commandBox:.*]] = fir.embox %[[commandCast]] : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK-NEXT: %[[exitstatBox:.*]] = fir.embox %arg1 : (!fir.ref<i[[DEFAULT_INTEGER_SIZE]]>) -> !fir.box<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-NEXT: %true = arith.constant true
! CHECK-NEXT: %[[wait:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[cmdmsg:.*]] = fir.absent !fir.box<none>
! CHECK: %[[sourceFileString:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileexitstat:.*]]>>
! CHECK-NEXT: %[[sourceLine:.*]] = arith.constant [[# @LINE - 9]] : i32
! CHECK-NEXT: %[[command:.*]] = fir.convert %[[commandBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK-NEXT: %[[exitstat:.*]] = fir.convert %[[exitstatBox]] : (!fir.box<i[[DEFAULT_INTEGER_SIZE]]>) -> !fir.box<none>
! CHECK-NEXT: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileexitstat]]>>) -> !fir.ref<i8>
! CHECK-NEXT: %{{.*}} = fir.call @_FortranAGetEnvVariable(%[[command]], %[[wait]], %[[exitstat]], %true, %[[cmdmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
end subroutine command_and_exitstat_only

! CHECK-LABEL: func @_QPcommand_and_cmdstat_only(
! CHECK-SAME: %[[commandArg:.*]]: !fir.boxchar<1> {fir.bindc_command = "command"},
! CHECK-SAME: %[[cmdstatArg:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]> {fir.bindc_command = "cmdstat"}) {
subroutine command_and_cmdstat_only(command, cmdstat)
    character(len=32) :: command
    integer :: cmdstat
    call execute_command_line(command, CMDSTAT=cmdstat)
! CHECK: %[[commandUnbox:.*]]:2 = fir.unboxchar %[[commandArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[commandCast:.*]] = fir.convert %[[commandUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK-NEXT: %[[commandBox:.*]] = fir.embox %[[commandCast]] : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK-NEXT: %true = arith.constant true
! CHECK-NEXT: %[[wait:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[exitstat:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[cmdmsg:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[sourceFileString:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileexitstat:.*]]>>
! CHECK-NEXT: %[[sourceLine:.*]] = arith.constant [[# @LINE - 9]] : i32
! CHECK-NEXT: %[[command:.*]] = fir.convert %[[commandBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK-NEXT: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileexitstat]]>>) -> !fir.ref<i8>
! CHECK-32-NEXT: %[[cmdstat:.*]] = fir.call @_FortranAGetEnvVariable(%[[command]], %[[wait]], %[[exitstat]], %true, %[[cmdmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64-NEXT: %[[cmdstat32:.*]] = fir.call @_FortranAGetEnvVariable(%[[command]], %[[wait]], %[[exitstat]], %true, %[[cmdmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64: %[[cmdstat:.*]] = fir.convert %[[cmdstat32]] : (i32) -> i64
! CHECK: fir.store %[[cmdstat]] to %[[cmdstatArg]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
end subroutine command_and_cmdstat_only

! CHECK-LABEL: func @_QPcommand_and_cmdmsg_only(
! CHECK-SAME: %[[commandArg:.*]]: !fir.boxchar<1> {fir.bindc_command = "command"},
! CHECK-SAME: %[[cmdmsgArg:.*]]: !fir.boxchar<1> {fir.bindc_command = "cmdmsg"}) {
subroutine command_and_cmdmsg_only(command, cmdmsg)
    character(len=32) :: command, cmdmsg
    call execute_command_line(command, CMDMSG=cmdmsg)
! CHECK: %[[cmdmsgUnbox:.*]]:2 = fir.unboxchar %[[cmdmsgArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[cmdmsgCast:.*]] = fir.convert %[[cmdmsgUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK-NEXT: %[[commandUnbox:.*]]:2 = fir.unboxchar %[[commandArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[commandCast:.*]] = fir.convert %[[commandUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK-NEXT: %[[commandBox:.*]] = fir.embox %[[commandCast]] : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK-NEXT: %[[cmdmsgBox:.*]] = fir.embox %[[cmdmsgCast]] : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK-NEXT: %true = arith.constant true
! CHECK-NEXT: %[[wait:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[exitstat:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[sourceFileString:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileexitstat:.*]]>>
! CHECK-NEXT: %[[sourceLine:.*]] = arith.constant [[# @LINE - 11]] : i32
! CHECK-NEXT: %[[command:.*]] = fir.convert %[[commandBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK-NEXT: %[[cmdmsg:.*]] = fir.convert %[[cmdmsgBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK-NEXT: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileexitstat]]>>) -> !fir.ref<i8>
! CHECK-NEXT: %{{[0-9]+}} = fir.call @_FortranAGetEnvVariable(%[[command]], %[[wait]], %[[exitstat]], %true, %[[cmdmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-NEXT: return
end subroutine command_and_cmdmsg_only

! CHECK-LABEL: func @_QPall_arguments(
! CHECK-SAME: %[[commandArg:[^:]*]]: !fir.boxchar<1> {fir.bindc_command = "command"},
! CHECK-SAME: %[[waitArg:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_command = "wait", fir.optional},
! CHECK-SAME: %[[exitstatArg:[^:]*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]> {fir.bindc_command = "exitstat"},
! CHECK-SAME: %[[cmdstatArg:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]> {fir.bindc_command = "cmdstat"},
! CHECK-SAME: %[[cmdmsgArg:.*]]: !fir.boxchar<1> {fir.bindc_command = "cmdmsg"}) {
subroutine all_arguments(command, wait, exitstat, cmdstat, cmdmsg)
    character(len=32) :: command, wait, cmdmsg
    integer :: exitstat, cmdstat
    call execute_command_line(command, wait, exitstat, cmdstat, cmdmsg)
! CHECK: %[[cmdmsgUnbox:.*]]:2 = fir.unboxchar %[[cmdmsgArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[cmdmsgCast:.*]] = fir.convert %[[cmdmsgUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK-NEXT: %[[commandUnbox:.*]]:2 = fir.unboxchar %[[commandArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[commandCast:.*]] = fir.convert %[[commandUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK-NEXT: %[[waitUnbox:.*]]:2 = fir.unboxchar %[[waitArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[waitCast:.*]] = fir.convert %[[waitUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK-NEXT: %[[commandBoxed:.*]] = fir.embox %[[commandCast]] : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK-NEXT: %[[waitBoxed:.*]] = fir.embox %[[waitCast]] : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK-NEXT: %[[exitstatBoxed:.*]] = fir.embox %[[exitstatArg]] : (!fir.ref<i[[DEFAULT_INTEGER_SIZE]]>) -> !fir.box<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-NEXT: %[[cmdmsgBoxed:.*]] = fir.embox %[[cmdmsgCast]] : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK: %[[sourceFileString:.*]] = fir.address_of(@_QQcl.[[fileString:.*]]) : !fir.ref<!fir.char<1,[[fileStringexitstat:.*]]>>
! CHECK-NEXT: %[[sourceLine:.*]] = arith.constant [[# @LINE - 22]] : i32
! CHECK-NEXT: %[[command:.*]] = fir.convert %[[commandBoxed]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK-NEXT: %[[wait:.*]] = fir.convert %[[waitBoxed]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK-NEXT: %[[exitstat:.*]] = fir.convert %[[exitstatBoxed]] : (!fir.box<i[[DEFAULT_INTEGER_SIZE]]>) -> !fir.box<none>
! CHECK-NEXT: %[[cmdmsg:.*]] = fir.convert %[[cmdmsgBoxed]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK-NEXT: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[fileStringexitstat]]>>) -> !fir.ref<i8>
! CHECK-32-NEXT: %[[cmdstat:.*]] = fir.call @_FortranAGetEnvVariable(%[[command]], %[[wait]], %[[exitstat]], %[[cmdmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64-NEXT: %[[cmdstat32:.*]] = fir.call @_FortranAGetEnvVariable(%[[command]], %[[wait]], %[[exitstat]], %[[cmdmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64: %[[cmdstat:.*]] = fir.convert %[[cmdstat32]] : (i32) -> i64
! CHECK: fir.store %[[cmdstat]] to %[[cmdstatArg]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
end subroutine all_arguments
