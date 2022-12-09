! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes=CHECK,CHECK-32 -DDEFAULT_INTEGER_SIZE=32 %s
! RUN: flang-new -fc1 -fdefault-integer-8 -emit-fir %s -o - | FileCheck --check-prefixes=CHECK,CHECK-64 -DDEFAULT_INTEGER_SIZE=64 %s

! CHECK-LABEL: func @_QPnumber_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"}) {
subroutine number_only(name)
    character(len=32) :: name
    call get_environment_variable(name)
! CHECK-NOT: fir.call @_FortranAEnvVariableValue
! CHECK-NOT: fir.call @_FortranAEnvVariableLength
! CHECK-NEXT: return
end subroutine number_only

! CHECK-LABEL: func @_QPname_and_value_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-SAME: %[[valueArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "value"}) {
subroutine name_and_value_only(name, value)
    character(len=32) :: name, value
    call get_environment_variable(name, value)
! CHECK: %[[nameUnbox:.*]]:2 = fir.unboxchar %[[nameArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[nameLength:.*]] = arith.constant 32 : index
! CHECK-NEXT: %[[valueUnbox:.*]]:2 = fir.unboxchar %[[valueArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[valueLength:.*]] = arith.constant 32 : index
! CHECK-NEXT: %[[nameBox:.*]] = fir.embox %[[nameUnbox]]#0 typeparams %[[nameLength]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-NEXT: %[[valueBox:.*]] = fir.embox %[[valueUnbox]]#0 typeparams %[[valueLength]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-NEXT: %true = arith.constant true
! CHECK-NEXT: %[[errmsg:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[sourceFileString:.*]] = fir.address_of(@_QQcl{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileLength:.*]]>>
! CHECK-NEXT: %[[sourceLine:.*]] = arith.constant [[# @LINE - 10]] : i32
! CHECK-NEXT: %[[name:.*]] = fir.convert %[[nameBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-NEXT: %[[value:.*]] = fir.convert %[[valueBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-NEXT: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileLength]]>>) -> !fir.ref<i8>
! CHECK-NEXT: %{{[0-9]+}} = fir.call @_FortranAEnvVariableValue(%[[name]], %[[value]], %true, %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-NEXT: return
end subroutine name_and_value_only

! CHECK-LABEL: func @_QPname_and_length_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-SAME: %[[lengthArg:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]> {fir.bindc_name = "length"}) {
subroutine name_and_length_only(name, length)
    character(len=32) :: name
    integer :: length
    call get_environment_variable(name, LENGTH=length)
! CHECK-NOT: fir.call @_FortranAEnvVariableValue
! CHECK: %[[nameUnbox:.*]]:2 = fir.unboxchar %[[nameArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[nameLength:.*]] = arith.constant 32 : index
! CHECK-NEXT: %[[nameBox:.*]] = fir.embox %0#0 typeparams %[[nameLength]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-NEXT: %true = arith.constant true
! CHECK: %[[sourceFileString:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileLength:.*]]>>
! CHECK-NEXT: %[[sourceLine:.*]] = arith.constant [[# @LINE - 7]] : i32
! CHECK-NEXT: %[[name:.*]] = fir.convert %[[nameBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-NEXT: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileLength]]>>) -> !fir.ref<i8>
! CHECK-32-NEXT: %[[length64:.*]] = fir.call @_FortranAEnvVariableLength(%[[name]], %true, %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, i1, !fir.ref<i8>, i32) -> i64
! CHECK-64-NEXT: %[[length:.*]] = fir.call @_FortranAEnvVariableLength(%[[name]], %true, %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, i1, !fir.ref<i8>, i32) -> i64
! CHECK-32-NEXT: %[[length:.*]] = fir.convert %[[length64]] : (i64) -> i32
! CHECK-NEXT: fir.store %[[length]] to %[[lengthArg]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
end subroutine name_and_length_only

! CHECK-LABEL: func @_QPname_and_status_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-SAME: %[[statusArg:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]> {fir.bindc_name = "status"}) {
subroutine name_and_status_only(name, status)
    character(len=32) :: name
    integer :: status
    call get_environment_variable(name, STATUS=status)
! CHECK: %[[nameUnbox:.*]]:2 = fir.unboxchar %[[nameArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[nameLength:.*]] = arith.constant 32 : index
! CHECK-NEXT: %[[nameBox:.*]] = fir.embox %[[nameUnbox]]#0 typeparams %[[nameLength]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-NEXT: %true = arith.constant true
! CHECK-NEXT: %[[value:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[errmsg:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[sourceFileString:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileLength:.*]]>>
! CHECK-NEXT: %[[sourceLine:.*]] = arith.constant [[# @LINE - 8]] : i32
! CHECK-NEXT: %[[name:.*]] = fir.convert %[[nameBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-NEXT: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileLength]]>>) -> !fir.ref<i8>
! CHECK-32-NEXT: %[[status:.*]] = fir.call @_FortranAEnvVariableValue(%[[name]], %[[value]], %true, %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64-NEXT: %[[status32:.*]] = fir.call @_FortranAEnvVariableValue(%[[name]], %[[value]], %true, %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64: %[[status:.*]] = fir.convert %[[status32]] : (i32) -> i64
! CHECK: fir.store %[[status]] to %[[statusArg]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
end subroutine name_and_status_only

! CHECK-LABEL: func @_QPname_and_trim_name_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-SAME: %[[trimNameArg:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "trim_name"}) {
subroutine name_and_trim_name_only(name, trim_name)
    character(len=32) :: name
    logical :: trim_name
    call get_environment_variable(name, TRIM_NAME=trim_name)
    ! CHECK-NOT: fir.call @_FortranAEnvVariableValue
    ! CHECK-NOT: fir.call @_FortranAEnvVariableLength
    ! CHECK-NEXT: return
end subroutine name_and_trim_name_only

! CHECK-LABEL: func @_QPname_and_errmsg_only(
! CHECK-SAME: %[[nameArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-SAME: %[[errmsgArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "errmsg"}) {
subroutine name_and_errmsg_only(name, errmsg)
    character(len=32) :: name, errmsg
    call get_environment_variable(name, ERRMSG=errmsg)
! CHECK: %[[errmsgUnbox:.*]]:2 = fir.unboxchar %[[errmsgArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[errmsgLength:.*]] = arith.constant 32 : index
! CHECK-NEXT: %[[nameUnbox:.*]]:2 = fir.unboxchar %[[nameArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[nameLength:.*]] = arith.constant 32 : index
! CHECK-NEXT: %[[nameBox:.*]] = fir.embox %[[nameUnbox]]#0 typeparams %[[nameLength]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-NEXT: %[[errmsgBox:.*]] = fir.embox %[[errmsgUnbox]]#0 typeparams %c32 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-NEXT: %true = arith.constant true
! CHECK-NEXT: %[[value:.*]] = fir.absent !fir.box<none>
! CHECK-NEXT: %[[sourceFileString:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,[[sourceFileLength:.*]]>>
! CHECK-NEXT: %[[sourceLine:.*]] = arith.constant [[# @LINE - 10]] : i32
! CHECK-NEXT: %[[name:.*]] = fir.convert %[[nameBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-NEXT: %[[errmsg:.*]] = fir.convert %[[errmsgBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-NEXT: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[sourceFileLength]]>>) -> !fir.ref<i8>
! CHECK-NEXT: %{{[0-9]+}} = fir.call @_FortranAEnvVariableValue(%[[name]], %[[value]], %true, %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-NEXT: return
end subroutine name_and_errmsg_only

! CHECK-LABEL: func @_QPall_arguments(
! CHECK-SAME: %[[nameArg:[^:]*]]: !fir.boxchar<1> {fir.bindc_name = "name"},
! CHECK-SAME: %[[valueArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "value"},
! CHECK-SAME: %[[lengthArg:[^:]*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]> {fir.bindc_name = "length"},
! CHECK-SAME: %[[statusArg:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]> {fir.bindc_name = "status"},
! CHECK-SAME: %[[trimNameArg:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "trim_name"},
! CHECK-SAME: %[[errmsgArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "errmsg"}) {
subroutine all_arguments(name, value, length, status, trim_name, errmsg)
    character(len=32) :: name, value, errmsg
    integer :: length, status
    logical :: trim_name
    call get_environment_variable(name, value, length, status, trim_name, errmsg)
! CHECK: %[[errmsgUnbox:.*]]:2 = fir.unboxchar %[[errmsgArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[errmsgLength:.*]] = arith.constant 32 : index
! CHECK-NEXT: %[[nameUnbox:.*]]:2 = fir.unboxchar %[[nameArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[nameLength:.*]] = arith.constant 32 : index
! CHECK-NEXT: %[[valueUnbox:.*]]:2 = fir.unboxchar %[[valueArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-NEXT: %[[valueLength:.*]] = arith.constant 32 : index
! CHECK-NEXT: %[[nameBoxed:.*]] = fir.embox %[[nameUnbox]]#0 typeparams %[[nameLength]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-NEXT: %[[valueBoxed:.*]] = fir.embox %[[valueUnbox]]#0 typeparams %[[valueLength]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-NEXT: %[[errmsgBoxed:.*]] = fir.embox %[[errmsgUnbox]]#0 typeparams %[[errmsgLength]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK:      %[[trimName:.*]] = fir.if %{{.*}} -> (i1) {
! CHECK-NEXT:   %[[trimNameLoaded:.*]] = fir.load %[[trimNameArg]] : !fir.ref<!fir.logical<4>>
! CHECK-NEXT:   %[[trimCast:.*]] = fir.convert %[[trimNameLoaded]] : (!fir.logical<4>) -> i1
! CHECK-NEXT:   fir.result %[[trimCast]] : i1
! CHECK-NEXT: } else {
! CHECK-NEXT:   %[[trueVal:.*]] = arith.constant true
! CHECK-NEXT:   fir.result %[[trueVal]] : i1
! CHECK-NEXT: }
! CHECK: %[[sourceFileString:.*]] = fir.address_of(@_QQcl.[[fileString:.*]]) : !fir.ref<!fir.char<1,[[fileStringLength:.*]]>>
! CHECK-NEXT: %[[sourceLine:.*]] = arith.constant [[# @LINE - 19]] : i32
! CHECK-NEXT: %[[name:.*]] = fir.convert %[[nameBoxed]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-NEXT: %[[value:.*]] = fir.convert %[[valueBoxed]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-NEXT: %[[errmsg:.*]] = fir.convert %[[errmsgBoxed]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-NEXT: %[[sourceFile:.*]] = fir.convert %[[sourceFileString]] : (!fir.ref<!fir.char<1,[[fileStringLength]]>>) -> !fir.ref<i8>
! CHECK-32-NEXT: %[[status:.*]] = fir.call @_FortranAEnvVariableValue(%[[name]], %[[value]], %[[trimName]], %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64-NEXT: %[[status32:.*]] = fir.call @_FortranAEnvVariableValue(%[[name]], %[[value]], %[[trimName]], %[[errmsg]], %[[sourceFile]], %[[sourceLine]]) {{.*}}: (!fir.box<none>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-64: %[[status:.*]] = fir.convert %[[status32]] : (i32) -> i64
! CHECK: fir.store %[[status]] to %[[statusArg]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
! CHECK: %[[sourceFileString2:.*]] = fir.address_of(@_QQcl.[[fileString]]) : !fir.ref<!fir.char<1,[[fileStringLength]]>>
! CHECK-NEXT: %[[sourceLine2:.*]] = arith.constant [[# @LINE - 29]] : i32
! CHECK-NEXT: %[[name:.*]] = fir.convert %[[nameBoxed]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-NEXT: %[[sourceFile2:.*]] = fir.convert %[[sourceFileString2]] : (!fir.ref<!fir.char<1,[[fileStringLength]]>>) -> !fir.ref<i8>
! CHECK-32-NEXT: %[[result64:.*]] = fir.call @_FortranAEnvVariableLength(%[[name]], %[[trimName]], %[[sourceFile2]], %[[sourceLine2]]) {{.*}}: (!fir.box<none>, i1, !fir.ref<i8>, i32) -> i64
! CHECK-64-NEXT: %[[result:.*]] = fir.call @_FortranAEnvVariableLength(%[[name]], %[[trimName]], %[[sourceFile2]], %[[sourceLine2]]) {{.*}}: (!fir.box<none>, i1, !fir.ref<i8>, i32) -> i64
! CHECK-32: %[[result:.*]] = fir.convert %[[result64]] : (i64) -> i32
! CHECK-NEXT: fir.store %[[result]] to %[[lengthArg]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
end subroutine all_arguments
