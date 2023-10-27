! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes=CHECK,CHECK-32 -DDEFAULT_INTEGER_SIZE=32 %s
! RUN: flang-new -fc1 -fdefault-integer-8 -emit-fir %s -o - | FileCheck --check-prefixes=CHECK,CHECK-64 -DDEFAULT_INTEGER_SIZE=64 %s

! CHECK-LABEL: func @_QPnumber_only(
! CHECK-SAME: %[[num:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>{{.*}}) {
subroutine number_only(num)
    integer :: num
    call get_command_argument(num)
! CHECK-NOT: fir.call @_FortranAGetCommandArgument
! CHECK-NEXT: return
end subroutine number_only

! CHECK-LABEL: func @_QPnumber_and_value_only(
! CHECK-SAME: %[[num:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>{{.*}}, %[[value:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine number_and_value_only(num, value)
integer :: num
character(len=32) :: value
call get_command_argument(num, value)
! CHECK: %[[valueUnboxed:.*]]:2 = fir.unboxchar %[[value]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[valueCast:.*]] = fir.convert %[[valueUnboxed]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK-NEXT: %[[numUnbox:.*]] = fir.load %[[num]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-NEXT: %[[valueBoxed:.*]] = fir.embox %[[valueCast]] : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK-NEXT: %[[length:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[errmsg:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[sourceFileString:.*]] = fir.address_of(@_QQcl{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileLength:.*]]>>
! CHECK-NEXT: %[[sourceLine:.*]] = arith.constant [[# @LINE - 8]] : i32
! CHECK-64-NEXT: %[[numCast:.*]] = fir.convert %[[numUnbox]] : (i64) -> i32
! CHECK-NEXT: %[[valueCast:.*]] = fir.convert %[[valueBoxed]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK-NEXT: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileLength]]>>) -> !fir.ref<i8>
! CHECK-32-NEXT: %{{[0-9]+}} = fir.call @_FortranAGetCommandArgument(%[[numUnbox]], %[[valueCast]], %[[length]], %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (i32, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64-NEXT: %{{[0-9]+}} = fir.call @_FortranAGetCommandArgument(%[[numCast]], %[[valueCast]], %[[length]], %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (i32, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
end subroutine number_and_value_only

! CHECK-LABEL: func @_QPall_arguments(
! CHECK-SAME: %[[num:[^:]*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>{{.*}}, %[[value:.*]]: !fir.boxchar<1>{{.*}}, %[[length:[^:]*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>{{.*}}, %[[status:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>{{.*}}, %[[errmsg:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine all_arguments(num, value, length, status, errmsg)
    integer :: num, length, status
    character(len=32) :: value, errmsg
    call get_command_argument(num, value, length, status, errmsg)
! CHECK: %[[errmsgUnboxed:.*]]:2 = fir.unboxchar %[[errmsg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[errmsgCast:.*]] = fir.convert %[[errmsgUnboxed]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK-NEXT: %[[valueUnboxed:.*]]:2 = fir.unboxchar %[[value]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[valueCast:.*]] = fir.convert %[[valueUnboxed]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK-NEXT: %[[numUnboxed:.*]] = fir.load %[[num]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-NEXT: %[[valueBoxed:.*]] = fir.embox %[[valueCast]] : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK-NEXT: %[[lengthBoxed:.*]] = fir.embox %[[length]] : (!fir.ref<i[[DEFAULT_INTEGER_SIZE]]>) -> !fir.box<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-NEXT: %[[errmsgBoxed:.*]] = fir.embox %[[errmsgCast]] : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK-NEXT: %[[sourceFileString:.*]] = fir.address_of(@_QQcl{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileLength:.*]]>>
! CHECK-NEXT: %[[sourceLine:.*]] = arith.constant [[# @LINE - 10]] : i32
! CHECK-64-NEXT: %[[numCast:.*]] = fir.convert %[[numUnboxed]] : (i64) -> i32
! CHECK-NEXT: %[[valueBuffer:.*]] = fir.convert %[[valueBoxed]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK-NEXT: %[[lengthBuffer:.*]] = fir.convert %[[lengthBoxed]] : (!fir.box<i[[DEFAULT_INTEGER_SIZE]]>) -> !fir.box<none>
! CHECK-NEXT: %[[errmsgBuffer:.*]] = fir.convert %[[errmsgBoxed]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK-NEXT: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileLength]]>>) -> !fir.ref<i8>
! CHECK-32-NEXT: %[[statusResult:.*]] = fir.call @_FortranAGetCommandArgument(%[[numUnboxed]], %[[valueBuffer]], %[[lengthBuffer]], %[[errmsgBuffer]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (i32, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64-NEXT: %[[statusResult32:.*]] = fir.call @_FortranAGetCommandArgument(%[[numCast]], %[[valueBuffer]], %[[lengthBuffer]], %[[errmsgBuffer]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (i32, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64: %[[statusResult:.*]] = fir.convert %[[statusResult32]] : (i32) -> i64
! CHECK: fir.store %[[statusResult]] to %[[status]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
end subroutine all_arguments

! CHECK-LABEL: func @_QPnumber_and_length_only(
! CHECK-SAME: %[[num:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>{{.*}}, %[[length:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>{{.*}}) {
subroutine number_and_length_only(num, length)
    integer :: num, length
    call get_command_argument(num, LENGTH=length)
! CHECK: %[[numLoaded:.*]] = fir.load %[[num]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-NEXT: %[[lengthBoxed:.*]] = fir.embox %[[length]] : (!fir.ref<i[[DEFAULT_INTEGER_SIZE]]>) -> !fir.box<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-NEXT: %[[value:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[errmsg:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[sourceFileString:.*]] = fir.address_of(@_QQcl{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileLength:.*]]>>
! CHECK-NEXT: %[[sourceLine:.*]] = arith.constant [[# @LINE - 6]] : i32
! CHECK-64: %[[numCast:.*]] = fir.convert %[[numLoaded]] : (i64) -> i32
! CHECK-NEXT: %[[lengthBuffer:.*]] = fir.convert %[[lengthBoxed]] : (!fir.box<i[[DEFAULT_INTEGER_SIZE]]>) -> !fir.box<none>
! CHECK-NEXT: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileLength]]>>) -> !fir.ref<i8>
! CHECK-32-NEXT: %{{.*}} = fir.call @_FortranAGetCommandArgument(%[[numLoaded]], %[[value]], %[[lengthBuffer]],  %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (i32, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64-NEXT: %{{.*}} = fir.call @_FortranAGetCommandArgument(%[[numCast]], %[[value]], %[[lengthBuffer]], %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (i32, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
end subroutine number_and_length_only

! CHECK-LABEL: func @_QPnumber_and_status_only(
! CHECK-SAME: %[[num:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>{{.*}}, %[[status:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>{{.*}}) {
subroutine number_and_status_only(num, status)
    integer :: num, status
    call get_command_argument(num, STATUS=status)
! CHECK: %[[numLoaded:.*]] = fir.load %[[num]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-NEXT: %[[value:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[length:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[errmsg:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[sourceFileString:.*]] = fir.address_of(@_QQcl{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileLength:.*]]>>
! CHECK-NEXT: %[[sourceLine:.*]] = arith.constant [[# @LINE - 6]] : i32
! CHECK-64: %[[numCast:.*]] = fir.convert %[[numLoaded]] : (i64) -> i32
! CHECK-NEXT: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileLength]]>>) -> !fir.ref<i8>
! CHECK-32-NEXT: %[[result:.*]] = fir.call @_FortranAGetCommandArgument(%[[numLoaded]], %[[value]], %[[length]],  %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (i32, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64-NEXT: %[[result32:.*]] = fir.call @_FortranAGetCommandArgument(%[[numCast]], %[[value]], %[[length]], %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (i32, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64: %[[result:.*]] = fir.convert %[[result32]] : (i32) -> i64
! CHECK-32: fir.store %[[result]] to %[[status]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
end subroutine number_and_status_only

! CHECK-LABEL: func @_QPnumber_and_errmsg_only(
! CHECK-SAME: %[[num:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>{{.*}}, %[[errmsg:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine number_and_errmsg_only(num, errmsg)
    integer :: num
    character(len=32) :: errmsg
    call get_command_argument(num, ERRMSG=errmsg)
! CHECK: %[[errmsgUnboxed:.*]]:2 = fir.unboxchar %[[errmsg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[errmsgCast:.*]] = fir.convert %[[errmsgUnboxed]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK-NEXT: %[[numUnboxed:.*]] = fir.load %[[num]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-NEXT: %[[errmsgBoxed:.*]] = fir.embox %[[errmsgCast]] : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK-NEXT: %[[value:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[length:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[sourceFileString:.*]] = fir.address_of(@_QQcl{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileLength:.*]]>>
! CHECK-NEXT: %[[sourceLine:.*]] = arith.constant [[# @LINE - 8]] : i32
! CHECK-64: %[[numCast:.*]] = fir.convert %[[numUnboxed]] : (i64) -> i32
! CHECK-NEXT: %[[errmsg:.*]] = fir.convert %[[errmsgBoxed]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK-NEXT: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileLength]]>>) -> !fir.ref<i8>
! CHECK-32-NEXT: %{{[0-9]+}} = fir.call @_FortranAGetCommandArgument(%[[numUnboxed]], %[[value]], %[[length]], %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (i32, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64-NEXT: %{{[0-9]+}} = fir.call @_FortranAGetCommandArgument(%[[numCast]], %[[value]], %[[length]], %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (i32, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
end subroutine number_and_errmsg_only
