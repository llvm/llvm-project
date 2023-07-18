! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes=CHECK,CHECK-32 -DDEFAULT_INTEGER_SIZE=32 %s
! RUN: flang-new -fc1 -fdefault-integer-8 -emit-fir %s -o - | FileCheck --check-prefixes=CHECK,CHECK-64 -DDEFAULT_INTEGER_SIZE=64 %s

! CHECK-LABEL: func @_QPname_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"}) {
subroutine name_only(name)
    character(len=32) :: name
    call get_environment_variable(name)
! CHECK-NOT: fir.call @_FortranAGetEnvVariable
! CHECK-NEXT: return
end subroutine name_only

! CHECK-LABEL: func @_QPname_and_value_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-SAME: %[[valueArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "value"}) {
subroutine name_and_value_only(name, value)
    character(len=32) :: name, value
    call get_environment_variable(name, value)
! CHECK: %[[nameUnbox:.*]]:2 = fir.unboxchar %[[nameArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[nameCast:.*]] = fir.convert %[[nameUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK-NEXT: %[[valueUnbox:.*]]:2 = fir.unboxchar %[[valueArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[valueCast:.*]] = fir.convert %[[valueUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK-NEXT: %[[nameBox:.*]] = fir.embox %[[nameCast]] : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK-NEXT: %[[valueBox:.*]] = fir.embox %[[valueCast]] : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK-NEXT: %true = arith.constant true
! CHECK-NEXT: %[[length:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[errmsg:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[sourceFileString:.*]] = fir.address_of(@_QQcl{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileLength:.*]]>>
! CHECK-NEXT: %[[sourceLine:.*]] = arith.constant [[# @LINE - 11]] : i32
! CHECK-NEXT: %[[name:.*]] = fir.convert %[[nameBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK-NEXT: %[[value:.*]] = fir.convert %[[valueBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK-NEXT: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileLength]]>>) -> !fir.ref<i8>
! CHECK-NEXT: %{{[0-9]+}} = fir.call @_FortranAGetEnvVariable(%[[name]], %[[value]], %[[length]], %true, %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-NEXT: return
end subroutine name_and_value_only

! CHECK-LABEL: func @_QPname_and_length_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-SAME: %[[lengthArg:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]> {fir.bindc_name = "length"}) {
subroutine name_and_length_only(name, length)
    character(len=32) :: name
    integer :: length
    call get_environment_variable(name, LENGTH=length)
! CHECK: %[[nameUnbox:.*]]:2 = fir.unboxchar %[[nameArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[nameCast:.*]] = fir.convert %[[nameUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK-NEXT: %[[nameBox:.*]] = fir.embox %[[nameCast]] : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK-NEXT: %[[lengthBox:.*]] = fir.embox %arg1 : (!fir.ref<i[[DEFAULT_INTEGER_SIZE]]>) -> !fir.box<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-NEXT: %true = arith.constant true
! CHECK-NEXT: %[[value:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[errmsg:.*]] = fir.absent !fir.box<none>
! CHECK: %[[sourceFileString:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileLength:.*]]>>
! CHECK-NEXT: %[[sourceLine:.*]] = arith.constant [[# @LINE - 9]] : i32
! CHECK-NEXT: %[[name:.*]] = fir.convert %[[nameBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK-NEXT: %[[length:.*]] = fir.convert %[[lengthBox]] : (!fir.box<i[[DEFAULT_INTEGER_SIZE]]>) -> !fir.box<none>
! CHECK-NEXT: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileLength]]>>) -> !fir.ref<i8>
! CHECK-NEXT: %{{.*}} = fir.call @_FortranAGetEnvVariable(%[[name]], %[[value]], %[[length]], %true, %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
end subroutine name_and_length_only

! CHECK-LABEL: func @_QPname_and_status_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-SAME: %[[statusArg:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]> {fir.bindc_name = "status"}) {
subroutine name_and_status_only(name, status)
    character(len=32) :: name
    integer :: status
    call get_environment_variable(name, STATUS=status)
! CHECK: %[[nameUnbox:.*]]:2 = fir.unboxchar %[[nameArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[nameCast:.*]] = fir.convert %[[nameUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK-NEXT: %[[nameBox:.*]] = fir.embox %[[nameCast]] : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK-NEXT: %true = arith.constant true
! CHECK-NEXT: %[[value:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[length:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[errmsg:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[sourceFileString:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileLength:.*]]>>
! CHECK-NEXT: %[[sourceLine:.*]] = arith.constant [[# @LINE - 9]] : i32
! CHECK-NEXT: %[[name:.*]] = fir.convert %[[nameBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK-NEXT: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileLength]]>>) -> !fir.ref<i8>
! CHECK-32-NEXT: %[[status:.*]] = fir.call @_FortranAGetEnvVariable(%[[name]], %[[value]], %[[length]], %true, %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64-NEXT: %[[status32:.*]] = fir.call @_FortranAGetEnvVariable(%[[name]], %[[value]], %[[length]], %true, %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64: %[[status:.*]] = fir.convert %[[status32]] : (i32) -> i64
! CHECK: fir.store %[[status]] to %[[statusArg]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
end subroutine name_and_status_only

! CHECK-LABEL: func @_QPname_and_trim_name_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-32-SAME: %[[trimNameArg:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "trim_name"}) {
! CHECK-64-SAME: %[[trimNameArg:.*]]: !fir.ref<!fir.logical<8>> {fir.bindc_name = "trim_name"}) {
subroutine name_and_trim_name_only(name, trim_name)
    character(len=32) :: name
    logical :: trim_name
    call get_environment_variable(name, TRIM_NAME=trim_name)
    ! CHECK-NOT: fir.call @_FortranAGetEnvVariable
    ! CHECK-NEXT: return
end subroutine name_and_trim_name_only

! CHECK-LABEL: func @_QPname_and_errmsg_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-SAME: %[[errmsgArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "errmsg"}) {
subroutine name_and_errmsg_only(name, errmsg)
    character(len=32) :: name, errmsg
    call get_environment_variable(name, ERRMSG=errmsg)
! CHECK: %[[errmsgUnbox:.*]]:2 = fir.unboxchar %[[errmsgArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[errmsgCast:.*]] = fir.convert %[[errmsgUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK-NEXT: %[[nameUnbox:.*]]:2 = fir.unboxchar %[[nameArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[nameCast:.*]] = fir.convert %[[nameUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK-NEXT: %[[nameBox:.*]] = fir.embox %[[nameCast]] : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK-NEXT: %[[errmsgBox:.*]] = fir.embox %[[errmsgCast]] : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK-NEXT: %true = arith.constant true
! CHECK-NEXT: %[[value:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[length:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[sourceFileString:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileLength:.*]]>>
! CHECK-NEXT: %[[sourceLine:.*]] = arith.constant [[# @LINE - 11]] : i32
! CHECK-NEXT: %[[name:.*]] = fir.convert %[[nameBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK-NEXT: %[[errmsg:.*]] = fir.convert %[[errmsgBox]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK-NEXT: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileLength]]>>) -> !fir.ref<i8>
! CHECK-NEXT: %{{[0-9]+}} = fir.call @_FortranAGetEnvVariable(%[[name]], %[[value]], %[[length]], %true, %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-NEXT: return
end subroutine name_and_errmsg_only

! CHECK-LABEL: func @_QPall_arguments(
! CHECK-SAME: %[[nameArg:[^:]*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-SAME: %[[valueArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "value"},
! CHECK-SAME: %[[lengthArg:[^:]*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]> {fir.bindc_name = "length"},
! CHECK-SAME: %[[statusArg:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]> {fir.bindc_name = "status"},
! CHECK-32-SAME: %[[trimNameArg:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "trim_name"},
! CHECK-64-SAME: %[[trimNameArg:.*]]: !fir.ref<!fir.logical<8>> {fir.bindc_name = "trim_name"},
! CHECK-SAME: %[[errmsgArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "errmsg"}) {
subroutine all_arguments(name, value, length, status, trim_name, errmsg)
    character(len=32) :: name, value, errmsg
    integer :: length, status
    logical :: trim_name
    call get_environment_variable(name, value, length, status, trim_name, errmsg)
! CHECK: %[[errmsgUnbox:.*]]:2 = fir.unboxchar %[[errmsgArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[errmsgCast:.*]] = fir.convert %[[errmsgUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK-NEXT: %[[nameUnbox:.*]]:2 = fir.unboxchar %[[nameArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[nameCast:.*]] = fir.convert %[[nameUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK-NEXT: %[[valueUnbox:.*]]:2 = fir.unboxchar %[[valueArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[valueCast:.*]] = fir.convert %[[valueUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,32>>
! CHECK-NEXT: %[[nameBoxed:.*]] = fir.embox %[[nameCast]] : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK-NEXT: %[[valueBoxed:.*]] = fir.embox %[[valueCast]] : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK-NEXT: %[[lengthBoxed:.*]] = fir.embox %[[lengthArg]] : (!fir.ref<i[[DEFAULT_INTEGER_SIZE]]>) -> !fir.box<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK-NEXT: %[[errmsgBoxed:.*]] = fir.embox %[[errmsgCast]] : (!fir.ref<!fir.char<1,32>>) -> !fir.box<!fir.char<1,32>>
! CHECK:      %[[trimName:.*]] = fir.if %{{.*}} -> (i1) {
! CHECK-32-NEXT:   %[[trimNameLoaded:.*]] = fir.load %[[trimNameArg]] : !fir.ref<!fir.logical<4>>
! CHECK-64-NEXT:   %[[trimNameLoaded:.*]] = fir.load %[[trimNameArg]] : !fir.ref<!fir.logical<8>>
! CHECK-32-NEXT:   %[[trimCast:.*]] = fir.convert %[[trimNameLoaded]] : (!fir.logical<4>) -> i1
! CHECK-64-NEXT:   %[[trimCast:.*]] = fir.convert %[[trimNameLoaded]] : (!fir.logical<8>) -> i1
! CHECK-NEXT:   fir.result %[[trimCast]] : i1
! CHECK-NEXT: } else {
! CHECK-NEXT:   %[[trueVal:.*]] = arith.constant true
! CHECK-NEXT:   fir.result %[[trueVal]] : i1
! CHECK-NEXT: }
! CHECK: %[[sourceFileString:.*]] = fir.address_of(@_QQcl.[[fileString:.*]]) : !fir.ref<!fir.char<1,[[fileStringLength:.*]]>>
! CHECK-NEXT: %[[sourceLine:.*]] = arith.constant [[# @LINE - 22]] : i32
! CHECK-NEXT: %[[name:.*]] = fir.convert %[[nameBoxed]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK-NEXT: %[[value:.*]] = fir.convert %[[valueBoxed]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK-NEXT: %[[length:.*]] = fir.convert %[[lengthBoxed]] : (!fir.box<i[[DEFAULT_INTEGER_SIZE]]>) -> !fir.box<none>
! CHECK-NEXT: %[[errmsg:.*]] = fir.convert %[[errmsgBoxed]] : (!fir.box<!fir.char<1,32>>) -> !fir.box<none>
! CHECK-NEXT: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[fileStringLength]]>>) -> !fir.ref<i8>
! CHECK-32-NEXT: %[[status:.*]] = fir.call @_FortranAGetEnvVariable(%[[name]], %[[value]], %[[length]], %[[trimName]], %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64-NEXT: %[[status32:.*]] = fir.call @_FortranAGetEnvVariable(%[[name]], %[[value]], %[[length]], %[[trimName]], %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64: %[[status:.*]] = fir.convert %[[status32]] : (i32) -> i64
! CHECK: fir.store %[[status]] to %[[statusArg]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
end subroutine all_arguments
