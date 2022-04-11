! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes=CHECK,CHECK-32 -DDEFAULT_INTEGER_SIZE=32 %s
! RUN: flang-new -fc1 -fdefault-integer-8 -emit-fir %s -o - | FileCheck --check-prefixes=CHECK,CHECK-64 -DDEFAULT_INTEGER_SIZE=64 %s

! CHECK-LABEL: func @_QPnumber_only(
! CHECK-SAME: %[[num:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>{{.*}}) {
subroutine number_only(num)
    integer :: num
    call get_command_argument(num)
! CHECK-NOT: fir.call @_FortranAArgumentValue
! CHECK-NOT: fir.call @_FortranAArgumentLength
! CHECK-NEXT: return
end subroutine number_only

! CHECK-LABEL: func @_QPnumber_and_value_only(
! CHECK-SAME: %[[num:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>{{.*}}, %[[value:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine number_and_value_only(num, value)
integer :: num
character(len=32) :: value
call get_command_argument(num, value)
! CHECK: %[[valueUnboxed:.*]]:2 = fir.unboxchar %[[value]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[valueLength:.*]] = arith.constant 32 : index
! CHECK-NEXT: %[[numUnbox:.*]] = fir.load %[[num]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-NEXT: %[[valueBoxed:.*]] = fir.embox %[[valueUnboxed]]#0 typeparams %[[valueLength]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-NEXT: %[[errmsg:.*]] = fir.absent !fir.box<none>
! CHECK-64-NEXT: %[[numCast:.*]] = fir.convert %[[numUnbox]] : (i64) -> i32
! CHECK-NEXT: %[[valueCast:.*]] = fir.convert %[[valueBoxed]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-32-NEXT: %{{[0-9]+}} = fir.call @_FortranAArgumentValue(%[[numUnbox]], %[[valueCast]], %[[errmsg]]) : (i32, !fir.box<none>, !fir.box<none>) -> i32
! CHECK-64-NEXT: %{{[0-9]+}} = fir.call @_FortranAArgumentValue(%[[numCast]], %[[valueCast]], %[[errmsg]]) : (i32, !fir.box<none>, !fir.box<none>) -> i32
! CHECK-NOT: fir.call @_FortranAArgumentLength
end subroutine number_and_value_only

! CHECK-LABEL: func @_QPall_arguments(
! CHECK-SAME: %[[num:[^:]*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>{{.*}}, %[[value:.*]]: !fir.boxchar<1>{{.*}}, %[[length:[^:]*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>{{.*}}, %[[status:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>{{.*}}, %[[errmsg:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine all_arguments(num, value, length, status, errmsg)
    integer :: num, length, status
    character(len=32) :: value, errmsg
    call get_command_argument(num, value, length, status, errmsg)
! CHECK: %[[errmsgUnboxed:.*]]:2 = fir.unboxchar %[[errmsg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[errmsgLen:.*]] = arith.constant 32 : index
! CHECK-NEXT: %[[valueUnboxed:.*]]:2 = fir.unboxchar %[[value]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[valueLen:.*]] = arith.constant 32 : index
! CHECK-NEXT: %[[numUnboxed:.*]] = fir.load %[[num]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-NEXT: %[[valueBoxed:.*]] = fir.embox %[[valueUnboxed]]#0 typeparams %[[valueLen]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-NEXT: %[[errmsgBoxed:.*]] = fir.embox %[[errmsgUnboxed]]#0 typeparams %[[errmsgLen]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-64-NEXT: %[[numCast:.*]] = fir.convert %[[numUnboxed]] : (i64) -> i32
! CHECK-NEXT: %[[valueBuffer:.*]] = fir.convert %[[valueBoxed]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-NEXT: %[[errmsgBuffer:.*]] = fir.convert %[[errmsgBoxed]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-32-NEXT: %[[statusResult:.*]] = fir.call @_FortranAArgumentValue(%[[numUnboxed]], %[[valueBuffer]], %[[errmsgBuffer]]) : (i32, !fir.box<none>, !fir.box<none>) -> i32
! CHECK-64-NEXT: %[[statusResult32:.*]] = fir.call @_FortranAArgumentValue(%[[numCast]], %[[valueBuffer]], %[[errmsgBuffer]]) : (i32, !fir.box<none>, !fir.box<none>) -> i32
! CHECK-64: %[[statusResult:.*]] = fir.convert %[[statusResult32]] : (i32) -> i64
! CHECK: fir.store %[[statusResult]] to %[[status]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-64: %[[numCast:.*]] = fir.convert %[[numUnboxed]] : (i64) -> i32
! CHECK-32: %[[lengthResult64:.*]] = fir.call @_FortranAArgumentLength(%[[numUnboxed]]) : (i32) -> i64
! CHECK-64: %[[lengthResult:.*]] = fir.call @_FortranAArgumentLength(%[[numCast]]) : (i32) -> i64
! CHECK-32-NEXT: %[[lengthResult:.*]] = fir.convert %[[lengthResult64]] : (i64) -> i32
! CHECK-NEXT: fir.store %[[lengthResult]] to %[[length]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
end subroutine all_arguments

! CHECK-LABEL: func @_QPnumber_and_length_only(
! CHECK-SAME: %[[num:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>{{.*}}, %[[length:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>{{.*}}) {
subroutine number_and_length_only(num, length)
    integer :: num, length
    call get_command_argument(num, LENGTH=length)
! CHECK-NOT: fir.call @_FortranAArgumentValue
! CHECK: %[[numLoaded:.*]] = fir.load %[[num]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-64: %[[numCast:.*]] = fir.convert %[[numLoaded]] : (i64) -> i32
! CHECK-32: %[[result64:.*]] = fir.call @_FortranAArgumentLength(%[[numLoaded]]) : (i32) -> i64
! CHECK-64: %[[result:.*]] = fir.call @_FortranAArgumentLength(%[[numCast]]) : (i32) -> i64
! CHECK-32-NEXT: %[[result:.*]] = fir.convert %[[result64]] : (i64) -> i32
! CHECK-NEXT: fir.store %[[result]] to %[[length]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
end subroutine number_and_length_only

! CHECK-LABEL: func @_QPnumber_and_status_only(
! CHECK-SAME: %[[num:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>{{.*}}, %[[status:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>{{.*}}) {
subroutine number_and_status_only(num, status)
    integer :: num, status
    call get_command_argument(num, STATUS=status)
! CHECK: %[[numLoaded:.*]] = fir.load %[[num]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-NEXT: %[[value:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[errmsg:.*]] = fir.absent !fir.box<none>
! CHECK-64: %[[numCast:.*]] = fir.convert %[[numLoaded]] : (i64) -> i32
! CHECK-32-NEXT: %[[result:.*]] = fir.call @_FortranAArgumentValue(%[[numLoaded]], %[[value]], %[[errmsg]]) : (i32, !fir.box<none>, !fir.box<none>) -> i32
! CHECK-64-NEXT: %[[result32:.*]] = fir.call @_FortranAArgumentValue(%[[numCast]], %[[value]], %[[errmsg]]) : (i32, !fir.box<none>, !fir.box<none>) -> i32
! CHECK-64: %[[result:.*]] = fir.convert %[[result32]] : (i32) -> i64
! CHECK-32: fir.store %[[result]] to %[[status]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-NOT: fir.call @_FortranAArgumentLength
end subroutine number_and_status_only

! CHECK-LABEL: func @_QPnumber_and_errmsg_only(
! CHECK-SAME: %[[num:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>{{.*}}, %[[errmsg:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine number_and_errmsg_only(num, errmsg)
    integer :: num
    character(len=32) :: errmsg
    call get_command_argument(num, ERRMSG=errmsg)
! CHECK: %[[errmsgUnboxed:.*]]:2 = fir.unboxchar %[[errmsg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[errmsgLength:.*]] = arith.constant 32 : index
! CHECK-NEXT: %[[numUnboxed:.*]] = fir.load %[[num]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-NEXT: %[[errmsgBoxed:.*]] = fir.embox %[[errmsgUnboxed]]#0 typeparams %[[errmsgLength]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-NEXT: %[[value:.*]] = fir.absent !fir.box<none>
! CHECK-64: %[[numCast:.*]] = fir.convert %[[numUnboxed]] : (i64) -> i32
! CHECK-NEXT: %[[errmsg:.*]] = fir.convert %[[errmsgBoxed]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-32-NEXT: %{{[0-9]+}} = fir.call @_FortranAArgumentValue(%[[numUnboxed]], %[[value]], %[[errmsg]]) : (i32, !fir.box<none>, !fir.box<none>) -> i32
! CHECK-64-NEXT: %{{[0-9]+}} = fir.call @_FortranAArgumentValue(%[[numCast]], %[[value]], %[[errmsg]]) : (i32, !fir.box<none>, !fir.box<none>) -> i32
! CHECK-NOT: fir.call @_FortranAArgumentLength
end subroutine number_and_errmsg_only
